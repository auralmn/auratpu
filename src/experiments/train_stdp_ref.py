"""
STDP Hebbian Learning Training Script - Line-by-Line Text File Processing
This script processes a text file one line at a time for neuromorphic language learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import time
from typing import Dict, List, Optional
from collections import defaultdict


# Import the fixed STDP implementation (assuming it's in the same directory)
# If you have the classes in a separate file, import them here
# from stdp_fixed_code import NeuromorphicLanguageProcessor
from llm_phasic import NeuromorphicLanguageProcessor


class SimpleTokenizer:
    """Simple character-level tokenizer for text processing"""

    def __init__(self, max_vocab_size: int = 1000):
        self.max_vocab_size = max_vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'

    def build_vocab(self, text_file_path: str):
        """Build vocabulary from text file"""
        print(f"Building vocabulary from {text_file_path}...")

        char_counts = defaultdict(int)

        with open(text_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 10000 == 0:
                    print(f"  Processing line {line_num}...")

                line = line.strip()
                for char in line:
                    char_counts[char] += 1

        # Sort by frequency and take top chars
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        top_chars = sorted_chars[:self.max_vocab_size - 2]  # Reserve space for special tokens

        # Build mappings
        self.char_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_char = {0: self.pad_token, 1: self.unk_token}

        for idx, (char, count) in enumerate(top_chars):
            token_idx = idx + 2
            self.char_to_idx[char] = token_idx
            self.idx_to_char[token_idx] = char

        self.vocab_size = len(self.char_to_idx)

        print(f"Built vocabulary with {self.vocab_size} characters")
        print(f"Most frequent chars: {[char for char, _ in sorted_chars[:10]]}")

    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """Convert text to token indices"""
        tokens = []
        for char in text[:max_length]:
            tokens.append(self.char_to_idx.get(char, 1))  # 1 is UNK token

        # Pad if necessary
        while len(tokens) < max_length:
            tokens.append(0)  # 0 is PAD token

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Convert token indices back to text"""
        chars = []
        for token in tokens:
            if token == 0:  # PAD token
                break
            chars.append(self.idx_to_char.get(token, self.unk_token))
        return ''.join(chars)


class STDPTrainer:
    """Training manager for STDP-based neuromorphic language model"""

    def __init__(self,
                 model: 'NeuromorphicLanguageProcessor',
                 tokenizer: SimpleTokenizer,
                 device: str = 'cpu',
                 log_interval: int = 100):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.log_interval = log_interval

        # Training statistics
        self.stats = {
            'lines_processed': 0,
            'total_tokens': 0,
            'avg_spike_rates_l1': [],
            'avg_spike_rates_l2': [],
            'weight_changes_l1': [],
            'weight_changes_l2': [],
            'processing_times': []
        }

    def process_single_line(self, line: str, seq_length: int = 64) -> Dict:
        """Process a single line of text with STDP learning"""

        # Clean and prepare line
        line = line.strip()
        if len(line) == 0:
            return None

        # Tokenize
        tokens = self.tokenizer.encode(line, max_length=seq_length)

        # Convert to tensor
        token_tensor = torch.tensor([tokens], device=self.device)  # [1, seq_length]

        # Forward pass with STDP learning
        start_time = time.time()

        self.model.train()
        output = self.model(token_tensor, apply_stdp=True)

        process_time = time.time() - start_time

        # Collect statistics
        stats = {
            'line_length': len(line),
            'num_tokens': len([t for t in tokens if t != 0]),  # Non-padding tokens
            'spike_rate_l1': output['layer1_spikes'].mean().item(),
            'spike_rate_l2': output['layer2_spikes'].mean().item(),
            'processing_time': process_time,
            'logits_mean': output['logits'].mean().item(),
            'logits_std': output['logits'].std().item()
        }

        # Add weight change info if available
        if self.model.layer1.weight_changes:
            stats['weight_change_l1'] = self.model.layer1.weight_changes[-1]
        if self.model.layer2.weight_changes:
            stats['weight_change_l2'] = self.model.layer2.weight_changes[-1]

        return stats

    def train_from_file(self,
                        text_file_path: str,
                        seq_length: int = 512,
                        max_lines: Optional[int] = None,
                        save_checkpoint_every: int = 1000) -> Dict:
        """
        Train the model by processing text file line by line

        Args:
            text_file_path: Path to text file
            seq_length: Maximum sequence length for each line
            max_lines: Maximum number of lines to process (None for all)
            save_checkpoint_every: Save model checkpoint every N lines
        """

        print(f"Starting line-by-line training from {text_file_path}")
        print(f"Sequence length: {seq_length}")
        print(f"Device: {self.device}")
        print("-" * 60)

        # Reset statistics
        self.stats = {
            'lines_processed': 0,
            'total_tokens': 0,
            'avg_spike_rates_l1': [],
            'avg_spike_rates_l2': [],
            'weight_changes_l1': [],
            'weight_changes_l2': [],
            'processing_times': []
        }

        start_time = time.time()

        with open(text_file_path, 'r', encoding='utf-8') as f:

            for line_num, line in enumerate(f):

                # Check max lines limit
                if max_lines and line_num >= max_lines:
                    break

                # Process line
                line_stats = self.process_single_line(line, seq_length)

                if line_stats is None:  # Skip empty lines
                    continue

                # Update global statistics
                self.stats['lines_processed'] += 1
                self.stats['total_tokens'] += line_stats['num_tokens']
                self.stats['avg_spike_rates_l1'].append(line_stats['spike_rate_l1'])
                self.stats['avg_spike_rates_l2'].append(line_stats['spike_rate_l2'])
                self.stats['processing_times'].append(line_stats['processing_time'])

                if 'weight_change_l1' in line_stats:
                    self.stats['weight_changes_l1'].append(line_stats['weight_change_l1'])
                if 'weight_change_l2' in line_stats:
                    self.stats['weight_changes_l2'].append(line_stats['weight_change_l2'])

                # Logging
                if (line_num + 1) % self.log_interval == 0:
                    self._print_progress(line_num + 1, line_stats, start_time)

                # Save checkpoint
                if (line_num + 1) % save_checkpoint_every == 0:
                    self._save_checkpoint(line_num + 1)

        # Final statistics
        total_time = time.time() - start_time
        self._print_final_stats(total_time)

        return self.stats

    def _print_progress(self, line_num: int, line_stats: Dict, start_time: float):
        """Print training progress"""

        elapsed = time.time() - start_time
        lines_per_sec = line_num / elapsed if elapsed > 0 else 0

        recent_l1 = np.mean(self.stats['avg_spike_rates_l1'][-self.log_interval:])
        recent_l2 = np.mean(self.stats['avg_spike_rates_l2'][-self.log_interval:])

        recent_weight_l1 = 0.0
        recent_weight_l2 = 0.0
        if self.stats['weight_changes_l1']:
            recent_weight_l1 = np.mean(
                self.stats['weight_changes_l1'][-min(self.log_interval, len(self.stats['weight_changes_l1'])):])
        if self.stats['weight_changes_l2']:
            recent_weight_l2 = np.mean(
                self.stats['weight_changes_l2'][-min(self.log_interval, len(self.stats['weight_changes_l2'])):])

        print(f"Line {line_num:6d} | "
              f"Speed: {lines_per_sec:5.1f} lines/sec | "
              f"Spikes L1: {recent_l1:.4f} L2: {recent_l2:.4f} | "
              f"ΔW L1: {recent_weight_l1:.6f} L2: {recent_weight_l2:.6f} | "
              f"Logits: μ={line_stats['logits_mean']:6.3f} σ={line_stats['logits_std']:6.3f}")

    def _print_final_stats(self, total_time: float):
        """Print final training statistics"""

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        print(f"Total lines processed: {self.stats['lines_processed']:,}")
        print(f"Total tokens processed: {self.stats['total_tokens']:,}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Average speed: {self.stats['lines_processed'] / total_time:.2f} lines/sec")

        if self.stats['avg_spike_rates_l1']:
            print(f"\nSpike Rate Statistics:")
            print(f"  Layer 1: μ={np.mean(self.stats['avg_spike_rates_l1']):.4f} "
                  f"σ={np.std(self.stats['avg_spike_rates_l1']):.4f}")
            print(f"  Layer 2: μ={np.mean(self.stats['avg_spike_rates_l2']):.4f} "
                  f"σ={np.std(self.stats['avg_spike_rates_l2']):.4f}")

        if self.stats['weight_changes_l1']:
            print(f"\nWeight Change Statistics:")
            print(f"  Layer 1: μ={np.mean(self.stats['weight_changes_l1']):.6f} "
                  f"final={self.stats['weight_changes_l1'][-1]:.6f}")
            print(f"  Layer 2: μ={np.mean(self.stats['weight_changes_l2']):.6f} "
                  f"final={self.stats['weight_changes_l2'][-1]:.6f}")

        print(f"\nAverage processing time per line: {np.mean(self.stats['processing_times']):.4f} sec")

    def _save_checkpoint(self, line_num: int):
        """Save model checkpoint"""
        checkpoint_path = f"stdp_model_checkpoint_line_{line_num}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'line_num': line_num,
            'stats': self.stats,
            'tokenizer_vocab': {
                'char_to_idx': self.tokenizer.char_to_idx,
                'idx_to_char': self.tokenizer.idx_to_char,
                'vocab_size': self.tokenizer.vocab_size
            }
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")


def main():
    """Main training function"""

    # Configuration
    TEXT_FILE_PATH = "all_books_merged.txt"  # ← Change this to your text file path
    VOCAB_SIZE = 32000
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 64
    SEQ_LENGTH = 64
    MAX_LINES = None  # Set to number to limit training, None for all lines
    LOG_INTERVAL = 1
    SAVE_EVERY = 50

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")

    # Check if text file exists
    if not os.path.exists(TEXT_FILE_PATH):
        print(f"❌ Text file not found: {TEXT_FILE_PATH}")
        print("Please create a text file with training data, one sentence/line per line.")
        print("Example content:")
        print("  The quick brown fox jumps over the lazy dog.")
        print("  Natural language processing with neuromorphic computing.")
        print("  Spike timing dependent plasticity enables learning.")
        return

    # Build tokenizer
    print("Step 1: Building vocabulary...")
    tokenizer = SimpleTokenizer(max_vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(TEXT_FILE_PATH)

    # Create model (you need to import/include the NeuromorphicLanguageProcessor class)
    print("Step 2: Creating model...")

    # Import the fixed STDP implementation
    from llm_phasic import NeuromorphicLanguageProcessor

    model = NeuromorphicLanguageProcessor(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        seq_length=SEQ_LENGTH,
        device=device
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    print("Step 3: Setting up trainer...")
    trainer = STDPTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        log_interval=LOG_INTERVAL
    )

    # Start training
    print("Step 4: Starting line-by-line training...")
    print(f"Processing file: {TEXT_FILE_PATH}")

    try:
        stats = trainer.train_from_file(
            text_file_path=TEXT_FILE_PATH,
            seq_length=SEQ_LENGTH,
            max_lines=MAX_LINES,
            save_checkpoint_every=SAVE_EVERY
        )

        # Save final model
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'training_stats': stats,
            'model_config': {
                'vocab_size': tokenizer.vocab_size,
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'output_dim': OUTPUT_DIM,
                'seq_length': SEQ_LENGTH
            },
            'tokenizer_vocab': {
                'char_to_idx': tokenizer.char_to_idx,
                'idx_to_char': tokenizer.idx_to_char,
                'vocab_size': tokenizer.vocab_size
            }
        }

        torch.save(final_checkpoint, 'stdp_model_final.pt')
        print("✅ Final model saved as 'stdp_model_final.pt'")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saving current model state...")

        interrupted_checkpoint = {
            'model_state_dict': model.state_dict(),
            'training_stats': trainer.stats,
            'interrupted': True
        }
        torch.save(interrupted_checkpoint, 'stdp_model_interrupted.pt')
        print("✅ Interrupted model saved as 'stdp_model_interrupted.pt'")

    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()


def test_trained_model(checkpoint_path: str = 'stdp_model_final.pt'):
    """Test a trained model on sample text"""

    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Reconstruct tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.char_to_idx = checkpoint['tokenizer_vocab']['char_to_idx']
    tokenizer.idx_to_char = checkpoint['tokenizer_vocab']['idx_to_char']
    tokenizer.vocab_size = checkpoint['tokenizer_vocab']['vocab_size']

    # Reconstruct model
    device = 'cuda' if torch.cuda.is_available() else 'mps'

    from llm_phasic import NeuromorphicLanguageProcessor

    model_config = checkpoint['model_config']
    model = NeuromorphicLanguageProcessor(
        vocab_size=model_config['vocab_size'],
        embed_dim=model_config['embed_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        seq_length=model_config['seq_length'],
        device=device
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test on sample text
    test_texts = [
        "Hello world this is a test",
        "Neuromorphic computing with spikes",
        "The quick brown fox jumps over",
        "Machine learning and artificial intelligence"
    ]

    print("\nTesting trained model:")
    print("-" * 40)

    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer.encode(text, max_length=512)
            token_tensor = torch.tensor([tokens], device=device)

            output = model(token_tensor, apply_stdp=False)

            print(f"Input: '{text}'")
            print(f"  Spike rates: L1={output['layer1_spikes'].mean().item():.4f}, "
                  f"L2={output['layer2_spikes'].mean().item():.4f}")
            print(f"  Logits range: [{output['logits'].min().item():.3f}, {output['logits'].max().item():.3f}]")
            print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else 'stdp_model_final.pt'
        test_trained_model(checkpoint_path)
    else:
        # Training mode
        main()