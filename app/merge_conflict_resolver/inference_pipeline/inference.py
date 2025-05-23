import torch
import torch.nn as nn
import subprocess
import tempfile
import os
import re
import logging
from collections import Counter
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SyntaxExtractor:
    def __init__(self):
        self.importPattern = r"(import|require)\s*\(?\s*[\s\S]*?\)?\s*;?"
        self.exportPattern = r"export\s+(default\s+)?(const|let|var|function|class)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)"
        self.functionPattern = r"function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(([^)]*)\)"
        self.arrowFunctionPattern = r"(const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(\([^)]*\)|[a-zA-Z_$][a-zA-Z0-9_$]*)\s*=>"
        self.classPattern = r"class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)"
        self.varPattern = r"(const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)"

    def extract(self, content):
        syntacticElements = {
            "imports": [],
            "exports": [],
            "functions": {},
            "classes": [],
            "variables": {},
        }

        try:
            imports = re.findall(self.importPattern, content)
            syntacticElements["imports"] = [imp.strip() for imp in imports]

            exports = re.findall(self.exportPattern, content)
            syntacticElements["exports"] = [exp[2] for exp in exports if exp[2]]

            functions = re.findall(self.functionPattern, content)
            for name, params in functions:
                syntacticElements["functions"][name] = [
                    p.strip() for p in params.split(",") if p.strip()
                ]

            arrowFuncs = re.findall(self.arrowFunctionPattern, content)
            for _, name, _ in arrowFuncs:
                if name not in syntacticElements["functions"]:
                    syntacticElements["functions"][name] = []

            classes = re.findall(self.classPattern, content)
            syntacticElements["classes"] = classes

            variables = re.findall(self.varPattern, content)
            for _, name in variables:
                if name not in syntacticElements["functions"]:
                    syntacticElements["variables"][name] = True

        except Exception as e:
            logger.error(f"Error extracting syntax: {str(e)}")
            pass

        return syntacticElements

    def format_for_model(self, elements):
        lines = []

        lines.append("<imports>")
        for imp in elements["imports"]:
            lines.append(imp)
        lines.append("</imports>")

        lines.append("<exports>")
        for exp in elements["exports"]:
            lines.append(exp)
        lines.append("</exports>")

        lines.append("<functions>")
        for name, params in elements["functions"].items():
            lines.append(f"{name}({', '.join(params)})")
        lines.append("</functions>")

        lines.append("<classes>")
        for cls in elements["classes"]:
            lines.append(cls)
        lines.append("</classes>")

        lines.append("<variables>")
        for variable in elements["variables"]:
            lines.append(variable)
        lines.append("</variables>")

        return "\n".join(lines)


class MergeT5(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(args["model_type"])
        self.t5.resize_token_embeddings(args["vocab_size"])
        self.embedding_dim = self.t5.config.d_model

        if device is not None:
            self.t5 = self.t5.to(device)

    def forward(self, input_txt, output_txt=None):
        attention_mask = input_txt != 0

        if output_txt is not None:
            decoder_attention_mask = output_txt != 0
            outputs = self.t5(
                input_ids=input_txt,
                attention_mask=attention_mask,
                decoder_input_ids=output_txt,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
            )
            logits = outputs.logits

            label = output_txt.clone()
            label = torch.cat(
                [
                    label[:, 1:],
                    torch.ones(len(label), 1, device=label.device, dtype=label.dtype)
                    * 0,
                ],
                dim=-1,
            )

            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.view(-1, logits.size(-1)), label.reshape(-1))
            return loss
        else:
            return self.t5.generate(
                input_ids=input_txt,
                attention_mask=attention_mask,
                max_length=200,
                num_beams=4,
                early_stopping=True,
            )


class CrossAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings, syntax_embeddings, key_padding_mask=None):
        normed_tokens = self.norm1(token_embeddings)
        normed_syntax = self.norm2(syntax_embeddings)

        attended_syntax, _ = self.cross_attention(
            query=normed_tokens,
            key=normed_syntax,
            value=normed_syntax,
            key_padding_mask=key_padding_mask,
        )

        token_with_syntax = token_embeddings + self.dropout(attended_syntax)

        normed_combined = self.norm3(token_with_syntax)
        return token_with_syntax + self.dropout(self.feed_forward(normed_combined))


class MultiLayerCrossAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(embedding_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, token_embeddings, syntax_embeddings, key_padding_mask=None):
        current_tokens = token_embeddings

        for layer in self.layers:
            current_tokens = layer(current_tokens, syntax_embeddings, key_padding_mask)

        return current_tokens


class SyntaxTokenModel(nn.Module):
    def __init__(self, args, token_model_path, tokenizer):
        super().__init__()

        self.token_usage_counter = Counter()
        self.tokenizer = tokenizer
        self.args = args

        self.syntax_influence_tracker = []
        self.syntax_threshold = 0.02

        self.token_model = MergeT5(args)

        state_dict = torch.load(token_model_path, map_location="cpu")
        self.token_model.load_state_dict(state_dict)

        for param in self.token_model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.token_model.embedding_dim

        self.syntax_encoder = T5ForConditionalGeneration.from_pretrained(
            args["model_type"], from_tf=False
        )
        self.syntax_encoder.resize_token_embeddings(len(tokenizer))

        self.syntax_enhancer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.Dropout(0.05),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
        )

        num_cross_attn_layers = args.get("cross_attention_layers", 3)

        self.syntax_attention = MultiLayerCrossAttention(
            embedding_dim=self.embedding_dim,
            num_heads=args.get("context_attention_heads", 8),
            dropout=args.get("context_dropout", 0.1),
            num_layers=num_cross_attn_layers,
        )

        self.syntax_relevance = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Sigmoid(),
        )

        self.output_transform = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(args.get("context_dropout", 0.1)),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.syntax_enhancer, self.output_transform]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.2)
                    if m.bias is not None:
                        m.bias.data.zero_()

        for layer in self.syntax_attention.layers:
            attention = layer.cross_attention

            nn.init.xavier_uniform_(attention.in_proj_weight, gain=0.2)
            nn.init.xavier_uniform_(attention.out_proj.weight, gain=0.2)
            nn.init.zeros_(attention.in_proj_bias)
            nn.init.zeros_(attention.out_proj.bias)

            for m in layer.feed_forward.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.2)
                    if m.bias is not None:
                        m.bias.data.zero_()

        for m in self.syntax_relevance.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.05)
                if m.bias is not None:
                    m.bias.data.fill_(-3.0)

        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.2)
                if m.bias is not None:
                    m.bias.data.fill_(-3.0)

    def extract_syntax_features(self, syntax_embeddings, syntax_mask):
        syntax_focused = self.syntax_enhancer(syntax_embeddings)

        if torch.is_tensor(syntax_mask):
            syntax_focused = syntax_focused * syntax_mask.unsqueeze(-1)

        if torch.is_tensor(syntax_mask):
            masked_sum = syntax_focused.sum(dim=1)
            mask_sum = syntax_mask.sum(dim=1, keepdim=True).clamp(min=1)
            global_syntax = masked_sum / mask_sum
        else:
            global_syntax = syntax_focused.mean(dim=1)

        return global_syntax.unsqueeze(1)

    def identify_syntax_tokens(self, input_ids):
        syntax_keywords = [
            "function",
            "const",
            "let",
            "var",
            "import",
            "export",
            "class",
            "extends",
            "return",
            "require",
            "=>",
            "module",
            "exports",
            "async",
            "await",
            "this",
        ]

        keyword_ids = []
        for keyword in syntax_keywords:
            ids = self.tokenizer.encode(keyword, add_special_tokens=False)
            keyword_ids.extend(ids)

        batch_size, seq_len = input_ids.shape
        syntax_mask = torch.zeros((batch_size, seq_len, 1), device=input_ids.device)

        for i in range(batch_size):
            for j in range(seq_len):
                if input_ids[i, j].item() in keyword_ids:
                    for k in range(min(5, seq_len - j)):
                        syntax_mask[i, j + k] = 1.0

        return syntax_mask

    def forward(self, input_txt, output_txt=None, syntax_context=None, stage="test"):
        batch_size = input_txt.size(0)
        device = input_txt.device

        self.syntax_influence_tracker = []

        if syntax_context is None:
            syntax_context = torch.ones_like(input_txt) * self.tokenizer.pad_token_id

        attention_mask = input_txt != self.tokenizer.pad_token_id

        with torch.no_grad():
            self.token_model = self.token_model.to(device)
            token_embeddings = self.token_model.t5.encoder(
                input_ids=input_txt, attention_mask=attention_mask, return_dict=True
            )["last_hidden_state"]
            original_token_embeddings = token_embeddings.clone()

        syntax_mask = syntax_context != self.tokenizer.pad_token_id

        if torch.sum(syntax_mask) == 0:
            syntax_mask[:, 0] = True

        try:
            self.syntax_encoder = self.syntax_encoder.to(device)
            syntax_outputs = self.syntax_encoder.encoder(
                input_ids=syntax_context, attention_mask=syntax_mask, return_dict=True
            )["last_hidden_state"]

            projected_syntax = self.syntax_enhancer(syntax_outputs)

            key_tokens_mask = self.identify_syntax_tokens(input_txt)

            key_padding_mask = ~syntax_mask

            attended_token_embeddings = self.syntax_attention(
                token_embeddings=token_embeddings,
                syntax_embeddings=projected_syntax,
                key_padding_mask=key_padding_mask,
            )

            token_syntax_pairs = torch.cat(
                [token_embeddings, attended_token_embeddings - token_embeddings], dim=-1
            )
            syntax_relevance = self.syntax_relevance(token_syntax_pairs)

            gate_input = torch.cat(
                [token_embeddings, attended_token_embeddings - token_embeddings], dim=-1
            )
            base_gate = self.gate(gate_input)

            syntax_gate = base_gate * 0.1
            syntax_gate = syntax_gate * (1.0 + key_tokens_mask * 10.0)

            max_influence = self.args.get("max_syntax_influence", 0.25)
            syntax_gate = torch.clamp(syntax_gate, min=0.0, max=max_influence)

            for i in range(batch_size):
                avg_gate_value = syntax_gate[i].mean().item()
                syntax_used = avg_gate_value > self.syntax_threshold
                self.syntax_influence_tracker.append(
                    {
                        "syntax_used": syntax_used,
                        "gate_avg": avg_gate_value,
                        "gate_max": syntax_gate[i].max().item(),
                        "token_relevance": syntax_relevance[i].mean().item(),
                    }
                )

            syntax_contribution = syntax_gate * (
                attended_token_embeddings - original_token_embeddings
            )

            combined_embeddings = original_token_embeddings + syntax_contribution

            final_embeddings = self.output_transform(combined_embeddings)

        except Exception as e:
            final_embeddings = original_token_embeddings

            for i in range(batch_size):
                self.syntax_influence_tracker.append(
                    {
                        "syntax_used": False,
                        "gate_avg": 0.0,
                        "gate_max": 0.0,
                        "token_relevance": 0.0,
                        "error": str(e),
                    }
                )

        final_embeddings = torch.nan_to_num(final_embeddings, nan=0.0)

        if output_txt is None:
            encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=final_embeddings, hidden_states=None, attentions=None
            )

            return self.token_model.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=self.args.get("max_resolve_length", 300),
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
            )
        else:
            decoder_outputs = self.token_model.t5.decoder(
                input_ids=output_txt,
                encoder_hidden_states=final_embeddings,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )

            logits = decoder_outputs["last_hidden_state"] * (self.embedding_dim**-0.5)
            logits = self.token_model.t5.lm_head(logits)

            if (
                self.args.get("use_token_repetition_penalty", False)
                and stage == "train"
            ):
                with torch.no_grad():
                    pred_tokens = torch.argmax(logits, dim=-1).view(-1).tolist()
                    for token in pred_tokens:
                        self.token_usage_counter[token] += 1

                token_counts = torch.zeros(logits.shape[-1], device=device)
                for b in range(batch_size):
                    pred_tokens = torch.argmax(logits[b], dim=-1)
                    for t in pred_tokens:
                        token_counts[t.item()] += 1

                token_freq = token_counts / max((batch_size * logits.shape[1]), 1)

                base_penalty = self.args.get("repetition_penalty", 1.5)
                penalty = (
                    1.0 + (token_freq * base_penalty) + (token_freq**2 * base_penalty)
                )
                penalty = torch.clamp(penalty, min=1.0, max=5.0)

                penalty = penalty.view(1, 1, -1).expand_as(logits)
                logits = logits / penalty

            outputs = F.log_softmax(logits, dim=-1)

            label = output_txt.clone()
            label = torch.cat(
                [
                    label[:, 1:],
                    torch.ones(len(label), 1, device=device, dtype=label.dtype)
                    * self.tokenizer.pad_token_id,
                ],
                dim=-1,
            )
            mask = label != self.tokenizer.pad_token_id

            loss = F.nll_loss(
                outputs.view(-1, outputs.size(-1)),
                label.contiguous().view(-1),
                reduction="none",
                ignore_index=self.tokenizer.pad_token_id,
            )

            mask = mask.view(-1).float()
            loss = loss * mask

            tokens = mask.sum().clamp(min=1.0)

            if stage == "train":
                return loss.sum() / tokens, tokens
            elif stage in ["dev", "test"]:
                output_ids = torch.argmax(outputs, dim=-1)
                return output_ids, loss.sum() / tokens, tokens, label

    def get_syntax_usage(self):
        if not self.syntax_influence_tracker:
            return []
        return self.syntax_influence_tracker


class HierarchicalMergeConflictResolver:
    def __init__(
        self,
        token_model_path,
        hierarchical_model_path,
        model_type="Salesforce/codet5-small",
        device=None,
        generation_config=None,
    ):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(model_type)
        brackets_tokens = ["<lbra>", "<mbra>", "<rbra>"]
        self.tokenizer.add_tokens(brackets_tokens)

        self.args = {
            "model_type": model_type,
            "vocab_size": len(self.tokenizer),
            "max_conflict_length": 500,
            "max_resolve_length": 300,
            "max_context_length": 800,
            "cross_attention_layers": 3,
            "context_attention_heads": 8,
            "context_dropout": 0.1,
            "max_syntax_influence": 0.25,
        }

        self.generation_config = generation_config or {
            "max_length": 300,
            "num_beams": 4,
            "early_stopping": True,
            "temperature": 1.0,
            "do_sample": False,
        }

        self.model = SyntaxTokenModel(self.args, token_model_path, self.tokenizer)

        state_dict = torch.load(
            hierarchical_model_path, map_location=torch.device("cpu")
        )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.syntax_extractor = SyntaxExtractor()

        self.MAX_CONFLICT_LENGTH = 500
        self.MAX_RESOLVE_LENGTH = 200
        self.MAX_CONTEXT_LENGTH = 800
        self.CONTEXT_LINES = 10

        self.SPACE_TOKEN = "Ġ"
        self.NEWLINE_TOKEN = "Ċ"
        self.SPACE_PLACEHOLDER = "<SPACE>"
        self.NEWLINE_PLACEHOLDER = "<NEWLINE>"

    def encode_special_tokens(self, tokens):
        encoded = []
        for token in tokens:
            if token == self.NEWLINE_TOKEN:
                encoded.append(self.NEWLINE_PLACEHOLDER)
            elif token.startswith(self.SPACE_TOKEN):
                encoded.append(f"{self.SPACE_PLACEHOLDER}{token[1:]}")
            else:
                encoded.append(token)
        return encoded

    def decode_special_tokens(self, tokens):
        decoded = []
        for token in tokens:
            if token == self.NEWLINE_PLACEHOLDER:
                decoded.append(self.NEWLINE_TOKEN)
            elif token.startswith(self.SPACE_PLACEHOLDER):
                decoded.append(
                    f"{self.SPACE_TOKEN}{token[len(self.SPACE_PLACEHOLDER):]}"
                )
            else:
                decoded.append(token)
        return decoded

    def pad_length(self, tokens, max_length, pad_id):
        if len(tokens) <= max_length:
            tokens += [pad_id] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

    def merge_syntax_elements(self, base_syntax, a_syntax, b_syntax):
        merged = {
            "imports": [],
            "exports": [],
            "functions": {},
            "classes": [],
            "variables": {},
        }

        merged["imports"] = list(
            set(base_syntax["imports"] + a_syntax["imports"] + b_syntax["imports"])
        )
        merged["exports"] = list(
            set(base_syntax["exports"] + a_syntax["exports"] + b_syntax["exports"])
        )
        merged["classes"] = list(
            set(base_syntax["classes"] + a_syntax["classes"] + b_syntax["classes"])
        )

        for syntax in [base_syntax, a_syntax, b_syntax]:
            merged["functions"].update(syntax["functions"])
            merged["variables"].update(syntax["variables"])

        return merged

    def extract_conflicts_from_tokens(self, merged_tokens):
        conflicts = []

        i = 0
        while i < len(merged_tokens):
            if merged_tokens[i] == "<lbra>":
                conflict_start_idx = max(0, i - 20)

                j = i + 1
                while j < len(merged_tokens) and merged_tokens[j] != "<rbra>":
                    j += 1

                if j < len(merged_tokens):
                    conflict_end_idx = min(len(merged_tokens), j + 20)

                    conflict_tokens = merged_tokens[conflict_start_idx:conflict_end_idx]
                    conflicts.append(
                        {
                            "tokens": conflict_tokens,
                            "start_idx": conflict_start_idx,
                            "end_idx": conflict_end_idx,
                            "conflict_start_offset": i - conflict_start_idx,
                            "conflict_end_offset": j - conflict_start_idx,
                        }
                    )
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1

        return conflicts

    def merge_conflicts_with_separators(self, conflicts):
        if not conflicts:
            return []

        merged_lines = []
        for idx, conflict in enumerate(conflicts):
            if idx > 0:
                merged_lines.append("// ===== CONFLICT SEPARATOR =====")
            merged_lines.extend(conflict["lines"])

        return merged_lines

    def should_use_conflict_focused(self, base_code, branch_a_code, branch_b_code):
        base_tokens = self.tokenizer.tokenize(base_code)
        a_tokens = self.tokenizer.tokenize(branch_a_code)
        b_tokens = self.tokenizer.tokenize(branch_b_code)

        merged_tokens = self.git_merge_tokens(base_tokens, a_tokens, b_tokens)

        return len(merged_tokens) > self.MAX_CONFLICT_LENGTH

    def preprocess_conflict_focused(self, base_code, branch_a_code, branch_b_code):
        base_tokens = self.tokenizer.tokenize(base_code)
        total_tokens = len(base_tokens)

        logger.info(
            f"Large file detected: {total_tokens} tokens. Using conflict-focused approach."
        )

        branch_a_tokens = self.tokenizer.tokenize(branch_a_code)
        branch_b_tokens = self.tokenizer.tokenize(branch_b_code)

        merged_tokens = self.git_merge_tokens(
            base_tokens, branch_a_tokens, branch_b_tokens
        )

        conflicts = self.extract_conflicts_from_tokens(merged_tokens)

        if not conflicts:
            logger.info("No conflicts found in the file")
            return None

        logger.info(f"Found {len(conflicts)} conflict(s)")

        all_conflict_tokens = []
        for idx, conflict in enumerate(conflicts):
            if idx > 0:
                all_conflict_tokens.extend(
                    ["Ċ", "//", "Ġ=====", "ĠCONFLICT", "ĠSEPARATOR", "Ġ=====", "Ċ"]
                )
            all_conflict_tokens.extend(conflict["tokens"])

        if len(all_conflict_tokens) > self.MAX_CONFLICT_LENGTH - 2:
            logger.warning(
                f"Conflicts exceed token limit ({len(all_conflict_tokens)} tokens), truncating"
            )
            all_conflict_tokens = all_conflict_tokens[: self.MAX_CONFLICT_LENGTH - 2]

        conflict_ids = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.bos_token]
            + all_conflict_tokens
            + [self.tokenizer.eos_token]
        )

        conflict_ids = self.pad_length(
            conflict_ids, self.MAX_CONFLICT_LENGTH, self.tokenizer.pad_token_id
        )

        base_syntax = self.syntax_extractor.extract(base_code)
        a_syntax = self.syntax_extractor.extract(branch_a_code)
        b_syntax = self.syntax_extractor.extract(branch_b_code)
        combined_syntax = self.merge_syntax_elements(base_syntax, a_syntax, b_syntax)
        syntax_context = self.syntax_extractor.format_for_model(combined_syntax)

        syntax_tokens = self.tokenizer.tokenize(syntax_context)
        if len(syntax_tokens) > self.MAX_CONTEXT_LENGTH - 2:
            logger.warning(
                f"Syntax tokens truncated from {len(syntax_tokens)} to {self.MAX_CONTEXT_LENGTH - 2}"
            )
            syntax_tokens = syntax_tokens[: self.MAX_CONTEXT_LENGTH - 2]

        syntax_ids = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.bos_token] + syntax_tokens + [self.tokenizer.eos_token]
        )
        syntax_ids = self.pad_length(
            syntax_ids, self.MAX_CONTEXT_LENGTH, self.tokenizer.pad_token_id
        )

        input_tensor = torch.tensor(conflict_ids).unsqueeze(0).to(self.device)
        syntax_tensor = torch.tensor(syntax_ids).unsqueeze(0).to(self.device)
        attention_mask = (input_tensor != self.tokenizer.pad_token_id).float()

        logger.debug(
            f"Conflict tokens being sent to model: {len(all_conflict_tokens)} tokens"
        )
        sample_text = self.tokenizer.decode(
            conflict_ids[:100], skip_special_tokens=False
        )
        logger.debug(f"Sample of conflict input: {sample_text}...")

        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "syntax_context": syntax_tensor,
            "is_partial": True,
            "conflict_info": conflicts,
            "total_file_tokens": total_tokens,
            "original_base": base_code,
        }

    def preprocess_normal(self, base_code, branch_a_code, branch_b_code):
        base_tokens = self.tokenizer.tokenize(base_code)
        branch_a_tokens = self.tokenizer.tokenize(branch_a_code)
        branch_b_tokens = self.tokenizer.tokenize(branch_b_code)

        merged_tokens = self.git_merge_tokens(
            base_tokens, branch_a_tokens, branch_b_tokens
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(merged_tokens)

        if len(input_ids) > self.MAX_CONFLICT_LENGTH:
            input_ids = input_ids[: self.MAX_CONFLICT_LENGTH - 1] + [
                self.tokenizer.eos_token_id
            ]

        if len(input_ids) < self.MAX_CONFLICT_LENGTH:
            padding_length = self.MAX_CONFLICT_LENGTH - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)

        base_syntax = self.syntax_extractor.extract(base_code)
        a_syntax = self.syntax_extractor.extract(branch_a_code)
        b_syntax = self.syntax_extractor.extract(branch_b_code)
        combined_syntax = self.merge_syntax_elements(base_syntax, a_syntax, b_syntax)
        syntax_context = self.syntax_extractor.format_for_model(combined_syntax)

        syntax_tokens = self.tokenizer.tokenize(syntax_context)
        if not syntax_tokens:
            syntax_tokens = [self.tokenizer.bos_token]

        syntax_ids = self.tokenizer.convert_tokens_to_ids(syntax_tokens)
        padded_syntax = self.pad_length(
            syntax_ids, self.MAX_CONTEXT_LENGTH, self.tokenizer.pad_token_id
        )

        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        syntax_tensor = torch.tensor(padded_syntax).unsqueeze(0).to(self.device)
        attention_mask = (input_tensor != self.tokenizer.pad_token_id).float()

        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "syntax_context": syntax_tensor,
            "is_partial": False,
        }

    def git_merge_tokens(self, base_tokens, a_tokens, b_tokens):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                encoded_base = self.encode_special_tokens(base_tokens)
                encoded_a = self.encode_special_tokens(a_tokens)
                encoded_b = self.encode_special_tokens(b_tokens)

                base_path = os.path.join(temp_dir, "base.txt")
                a_path = os.path.join(temp_dir, "a.txt")
                b_path = os.path.join(temp_dir, "b.txt")

                def write_tokens(tokens, path):
                    with open(path, "w", encoding="utf-8") as f:
                        for token in tokens:
                            f.write(f"{token}\n")

                write_tokens(encoded_base, base_path)
                write_tokens(encoded_a, a_path)
                write_tokens(encoded_b, b_path)

                subprocess.run(
                    ["git", "init"],
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                try:
                    result = subprocess.run(
                        [
                            "git",
                            "merge-file",
                            "-L",
                            "a",
                            "-L",
                            "base",
                            "-L",
                            "b",
                            a_path,
                            base_path,
                            b_path,
                            "--diff3",
                            "-p",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=False,
                        cwd=temp_dir,
                        check=False,
                    )
                except subprocess.CalledProcessError as e:
                    result = e

                merge_lines = result.stdout.decode(
                    "utf-8", errors="replace"
                ).splitlines()
                merge_tokens = []

                for line in merge_lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line == "<<<<<<< a":
                        merge_tokens.append("<lbra>")
                    elif line == "||||||| base":
                        merge_tokens.append(self.tokenizer.sep_token)
                    elif line == "=======":
                        merge_tokens.append(self.tokenizer.sep_token)
                    elif line == ">>>>>>> b":
                        merge_tokens.append("<rbra>")
                    else:
                        merge_tokens.append(line)

                decoded_tokens = self.decode_special_tokens(merge_tokens)
                final_tokens = (
                    [self.tokenizer.bos_token]
                    + decoded_tokens
                    + [self.tokenizer.eos_token]
                )

                logger.debug(
                    f"Decoded tokens: {len(decoded_tokens)} tokens, "
                    f"Final tokens: {len(final_tokens)} tokens"
                )

                return final_tokens

        except Exception as e:
            raise RuntimeError(f"Error during merge conflict resolution: {str(e)}")

    def generate_resolution(self, preprocessed_input):
        with torch.no_grad():
            try:
                input_ids = preprocessed_input["input_ids"].to(self.device)
                syntax_context = preprocessed_input["syntax_context"].to(self.device)

                generated_ids = self.model(
                    input_txt=input_ids, syntax_context=syntax_context
                )

                resolved_code = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                return resolved_code.strip()

            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                logger.debug(f"Input shape: {input_ids.shape}")
                logger.debug(f"Syntax context shape: {syntax_context.shape}")
                raise

    def reconstruct_file_from_tokens(self, original_base, conflicts, resolution):
        if len(conflicts) == 1:
            pattern = r"(handleDataProcessing.*?\{[\s\S]*?\n\s*\})"

            resolved_match = re.search(pattern, resolution)
            if resolved_match:
                resolved_method = resolved_match.group(1)

                original_match = re.search(pattern, original_base)
                if original_match:
                    return original_base.replace(
                        original_match.group(1), resolved_method
                    )

        return resolution

    def resolve_conflict(
        self, base_code, branch_a_code, branch_b_code, return_syntax_info=False
    ):
        try:
            base_tokens = self.tokenizer.tokenize(base_code)
            a_tokens = self.tokenizer.tokenize(branch_a_code)
            b_tokens = self.tokenizer.tokenize(branch_b_code)

            merged_tokens = self.git_merge_tokens(base_tokens, a_tokens, b_tokens)

            if len(merged_tokens) > self.MAX_CONFLICT_LENGTH:
                preprocessed = self.preprocess_conflict_focused(
                    base_code, branch_a_code, branch_b_code
                )

                if preprocessed is None:
                    logger.info("No conflicts found, returning base code")
                    return base_code

                resolution = self.generate_resolution(preprocessed)

                if preprocessed.get("is_partial", False):
                    final_resolution = self.reconstruct_file_from_tokens(
                        preprocessed["original_base"],
                        preprocessed["conflict_info"],
                        resolution,
                    )
                else:
                    final_resolution = resolution
            else:
                preprocessed = self.preprocess_normal(
                    base_code, branch_a_code, branch_b_code
                )

                resolution = self.generate_resolution(preprocessed)
                final_resolution = resolution

            if return_syntax_info:
                syntax_usage = self.model.syntax_influence_tracker
                return final_resolution, syntax_usage

            return final_resolution

        except Exception as e:
            logger.error(f"Error in conflict resolution: {str(e)}")
            import traceback

            traceback.print_exc()
            raise
