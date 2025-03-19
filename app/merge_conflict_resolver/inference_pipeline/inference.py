import torch
import torch.nn as nn
import subprocess
import tempfile
import os
from transformers import RobertaTokenizer, T5ForConditionalGeneration


class MergeT5(nn.Module):
    def __init__(self, model_type="Salesforce/codet5-small"):
        super(MergeT5, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_type)
        self.embedding_dim = self.t5.config.hidden_size

    def generate(self, *args, **kwargs):
        return self.t5.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.t5(*args, **kwargs)


class MergeConflictResolver:
    def __init__(self, model_path, model_type="Salesforce/codet5-small"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_type)
        brackets_tokens = ["<lbra>", "<mbra>", "<rbra>"]
        self.tokenizer.add_tokens(brackets_tokens)
        self.model = MergeT5(model_type)
        self.model.t5.resize_token_embeddings(len(self.tokenizer))
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.MAX_CONFLICT_LENGTH = 500
        self.MAX_RESOLVE_LENGTH = 200

        # Special token mappings
        self.SPACE_TOKEN = "Ġ"
        self.NEWLINE_TOKEN = "Ċ"
        self.SPACE_PLACEHOLDER = "<SPACE>"
        self.NEWLINE_PLACEHOLDER = "<NEWLINE>"

    def encode_special_tokens(self, tokens):
        """Replace special tokens with placeholders for git merge"""
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
        """Restore special tokens from placeholders after git merge"""
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

    def preprocess_merge_conflict(self, base_code, branch_a_code, branch_b_code):
        """Preprocess merge conflict components into model-ready input"""
        # Tokenize input code
        base_tokens = self.tokenizer.tokenize(base_code)
        branch_a_tokens = self.tokenizer.tokenize(branch_a_code)
        branch_b_tokens = self.tokenizer.tokenize(branch_b_code)

        # Get merged tokens using diff3
        merged_tokens = self.git_merge_tokens(
            base_tokens, branch_a_tokens, branch_b_tokens
        )

        # Convert tokens to ids with proper formatting
        input_ids = self.tokenizer.convert_tokens_to_ids(merged_tokens)

        # Truncate if necessary
        if len(input_ids) > self.MAX_CONFLICT_LENGTH:
            input_ids = input_ids[: self.MAX_CONFLICT_LENGTH - 1] + [
                self.tokenizer.eos_token_id
            ]

        # Pad if necessary
        if len(input_ids) < self.MAX_CONFLICT_LENGTH:
            padding_length = self.MAX_CONFLICT_LENGTH - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)

        # Convert to tensor and create attention mask
        input_tensor = torch.tensor(input_ids).unsqueeze(0)
        attention_mask = (input_tensor != self.tokenizer.pad_token_id).float()

        return {"input_ids": input_tensor, "attention_mask": attention_mask}

    def git_merge_tokens(self, base_tokens, a_tokens, b_tokens):
        """Perform a three-way merge at the token level using git merge-file"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Encode special tokens before writing to files
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
                in_conflict = False
                current_section = None

                for line in merge_lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line == "<<<<<<< a":
                        in_conflict = True
                        current_section = "a"
                        merge_tokens.append("<lbra>")
                    elif line == "||||||| base":
                        current_section = "base"
                        merge_tokens.append(self.tokenizer.sep_token)
                    elif line == "=======":
                        current_section = "b"
                        merge_tokens.append(self.tokenizer.sep_token)
                    elif line == ">>>>>>> b":
                        in_conflict = False
                        current_section = None
                        merge_tokens.append("<rbra>")
                    else:
                        merge_tokens.append(line)

                # Decode special tokens after merge
                decoded_tokens = self.decode_special_tokens(merge_tokens)
                final_tokens = (
                    [self.tokenizer.bos_token]
                    + decoded_tokens
                    + [self.tokenizer.eos_token]
                )

                return final_tokens

        except Exception as e:
            raise RuntimeError(f"Error during merge conflict resolution: {str(e)}")

    def generate_resolution(self, preprocessed_input):
        """Generate merge conflict resolution using trained model"""
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(
                    input_ids=preprocessed_input["input_ids"],
                    attention_mask=preprocessed_input["attention_mask"],
                    max_length=self.MAX_RESOLVE_LENGTH,
                    num_return_sequences=1,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                )

                resolved_code = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                return resolved_code.strip()

            except Exception as e:
                print(f"Generation error: {str(e)}")
                print(f"Input shape: {preprocessed_input['input_ids'].shape}")
                print(
                    f"First few input tokens: {self.tokenizer.decode(preprocessed_input['input_ids'][0][:10])}"
                )
                raise

    def resolve_conflict(self, base_code, branch_a_code, branch_b_code):
        """Convenience method to resolve a merge conflict in one step"""
        preprocessed = self.preprocess_merge_conflict(
            base_code, branch_a_code, branch_b_code
        )
        return self.generate_resolution(preprocessed)
