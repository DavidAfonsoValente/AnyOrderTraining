"""
data/visualize_examples.py
Visualise masked training examples for each of the five AOMT training methods.

Renders to terminal (rich-formatted) and optionally saves to PNG via matplotlib.
Run:
    python aomt/data/visualize_examples.py
    python aomt/data/visualize_examples.py --png_out ./figures/training_examples.png
    python aomt/data/visualize_examples.py --tokenizer ./models/llada2-mini-sep
"""

import argparse
import sys
import numpy as np

# ---- Example trajectory (shared across all visualisations) ------------------

EXAMPLE_UNITS = [
    {"type": "obs", "text": "You are in a kitchen. Task: put apple on table.\n> look"},
    {"type": "act", "text": "Think: I need to find an apple.\nAct: go to counter 1"},
    {"type": "obs", "text": "You see an apple on the counter."},
    {"type": "act", "text": "Think: Take the apple.\nAct: take apple from counter 1"},
    {"type": "obs", "text": "You are now holding the apple."},
    {"type": "act", "text": "Think: Go to table.\nAct: go to table 1"},
    {"type": "obs", "text": "You are at the table."},
]

UNIT_LABELS = [f"O{i//2}" if u["type"] == "obs" else f"A{i//2}"
               for i, u in enumerate(EXAMPLE_UNITS)]

# ---- Terminal visualisation (no dependencies) --------------------------------

ANSI = {
    "reset":       "\033[0m",
    "bold":        "\033[1m",
    "obs_clean":   "\033[48;5;27m\033[97m",    # blue bg, white text
    "act_clean":   "\033[48;5;28m\033[97m",    # green bg, white text
    "obs_masked":  "\033[48;5;240m\033[90m",   # grey bg, dark text (masked)
    "act_masked":  "\033[48;5;88m\033[91m",    # dark red bg, red text (masked)
    "prompt":      "\033[48;5;22m\033[97m",    # dark green bg (prompt context)
    "response":    "\033[48;5;88m\033[91m",    # dark red bg (masked response)
}


def render_unit_terminal(label: str, utype: str, masked: bool,
                          is_prompt: bool = False, is_response: bool = False) -> str:
    """Return ANSI-coloured unit label for terminal output."""
    if is_prompt:
        style = ANSI["prompt"]
        tag = "PROMPT"
    elif is_response:
        style = ANSI["response"]
        tag = "MASKED"
    elif masked:
        style = ANSI["obs_masked"] if utype == "obs" else ANSI["act_masked"]
        tag = "[MASK]"
    else:
        style = ANSI["obs_clean"] if utype == "obs" else ANSI["act_clean"]
        tag = "clean"

    return f"{style} {label}:{tag} {ANSI['reset']}"


def print_method(title: str, description: str, unit_states: list):
    """
    unit_states: list of ("label", "obs"|"act", state)
    state: "prompt" | "masked_response" | "masked" | "clean"
    """
    print(f"\n{ANSI['bold']}{'─'*64}{ANSI['reset']}")
    print(f"{ANSI['bold']}{title}{ANSI['reset']}")
    print(f"  {description}")
    print()

    row = "  "
    legend = []
    for label, utype, state in unit_states:
        is_prompt   = state == "prompt"
        is_response = state == "masked_response"
        masked      = state in ("masked", "masked_response")
        cell = render_unit_terminal(label, utype, masked, is_prompt, is_response)
        row += cell + " "
        legend.append((label, state))

    print(row)
    print()

    # Legend
    for label, state in legend:
        marker = {
            "prompt":          "→ bidirectional context (always clean)",
            "masked_response": "→ masked response (loss computed here)",
            "masked":          "→ masked target (loss computed here)",
            "clean":           "→ bidirectional context (clean)",
        }[state]
        print(f"    {label}: {marker}")


def visualize_terminal():
    print(f"\n{ANSI['bold']}{'='*64}{ANSI['reset']}")
    print(f"{ANSI['bold']}AOMT Training Example Visualisation{ANSI['reset']}")
    print(f"Trajectory: O0 A0 O1 A1 O2 A2 O3  (4 actions, 5 observations)")
    print()
    print("Colour key:")
    print(f"  {ANSI['prompt']} Xu:PROMPT {ANSI['reset']}  = clean context (never in loss)")
    print(f"  {ANSI['obs_clean']} Ox:clean  {ANSI['reset']}  = clean obs context")
    print(f"  {ANSI['act_clean']} Ax:clean  {ANSI['reset']}  = clean act context")
    print(f"  {ANSI['response']} Xu:MASKED {ANSI['reset']}  = masked → loss here")

    # ---- Method 1: Standard SFT (datapoint for A1) --------------------------
    # Prompt = O0 A0 O1; Response = A1 (masked)
    print_method(
        "1. Standard SFT  (datapoint predicting A1)",
        "Prompt = full causal history up to O1. Response = A1 (deterministically masked).\n"
        "  Future O2, A2, O3 absent from sequence — causal constraint via data.",
        [
            ("O0", "obs", "prompt"),
            ("A0", "act", "prompt"),
            ("O1", "obs", "prompt"),
            ("A1", "act", "masked_response"),
            # O2, A2, O3 not present
        ]
    )

    # ---- Method 2: Prefix SFT Stage 1 (predicting O2 from O1,A1) -----------
    print_method(
        "2. Prefix SFT Stage 1  (datapoint predicting O2 from local context)",
        "Prompt = (O1, A1) LOCAL context only — full history excluded intentionally.\n"
        "  Response = O2 (next observation, deterministically masked).",
        [
            ("O1", "obs", "prompt"),
            ("A1", "act", "prompt"),
            ("O2", "obs", "masked_response"),
        ]
    )

    # ---- Method 3: Prefix SFT Stage 2 (same as Standard SFT) ---------------
    print_method(
        "3. Prefix SFT Stage 2  (identical structure to Standard SFT)",
        "Same data format as Standard SFT. Initialised from Stage 1 checkpoint.\n"
        "  Restores action-prediction after world-model pretraining.",
        [
            ("O0", "obs", "prompt"),
            ("A0", "act", "prompt"),
            ("O1", "obs", "prompt"),
            ("A1", "act", "masked_response"),
        ]
    )

    # ---- Method 4: AOMT-Action-Only (example: A0 and A2 masked) ------------
    print_method(
        "4. AOMT-Action-Only  (example mask: A0 and A2 masked, A1 clean)",
        "Full trajectory in sequence. Action units masked via Bernoulli(p=0.25).\n"
        "  Obs units always clean. Future obs (O1 when predicting A0) = bidirectional context.",
        [
            ("O0", "obs", "clean"),
            ("A0", "act", "masked"),    # masked this step
            ("O1", "obs", "clean"),    # O1 available as context for A0!
            ("A1", "act", "clean"),
            ("O2", "obs", "clean"),
            ("A2", "act", "masked"),   # masked this step
            ("O3", "obs", "clean"),
        ]
    )

    # ---- Method 5: AOMT-Mixed (example: A0, O2 masked) ---------------------
    print_method(
        "5. AOMT-Mixed  (example mask: A0 and O2 masked)",
        "Full trajectory. Both obs and act units masked via Bernoulli(p=0.25).\n"
        "  Joint world-model + policy training. No separate stages.",
        [
            ("O0", "obs", "clean"),
            ("A0", "act", "masked"),   # act masked → policy training signal
            ("O1", "obs", "clean"),
            ("A1", "act", "clean"),
            ("O2", "obs", "masked"),   # obs masked → world-model training signal
            ("A2", "act", "clean"),
            ("O3", "obs", "clean"),
        ]
    )

    print(f"\n{ANSI['bold']}{'='*64}{ANSI['reset']}\n")
    print("Key differences across methods:")
    print("  • Std SFT vs AOMT-Act: SAME masking granularity (unit-level),")
    print("    DIFFERENT context (causal-only vs bidirectional with future obs).")
    print("  • AOMT-Act vs AOMT-Mixed: SAME context, DIFFERENT eligible units")
    print("    (actions-only vs all units).")
    print("  • All five use compute_unit_mask_loss — no 1/p_mask weighting.")
    print()


# ---- Matplotlib visualisation (PNG) -----------------------------------------

def visualize_png(output_path: str, tokenizer=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    methods = [
        {
            "title": "1. Standard SFT\n(datapoint for A₁)",
            "units": [
                ("O₀", "obs", "prompt"), ("A₀", "act", "prompt"),
                ("O₁", "obs", "prompt"), ("A₁", "act", "response"),
                # O₂ A₂ O₃ absent
            ],
        },
        {
            "title": "2. Prefix SFT Stage 1\n(predicting O₂)",
            "units": [
                ("O₁", "obs", "prompt"), ("A₁", "act", "prompt"),
                ("O₂", "obs", "response"),
            ],
        },
        {
            "title": "3. Prefix SFT Stage 2\n(same structure as Std SFT)",
            "units": [
                ("O₀", "obs", "prompt"), ("A₀", "act", "prompt"),
                ("O₁", "obs", "prompt"), ("A₁", "act", "response"),
            ],
        },
        {
            "title": "4. AOMT-Action-Only\n(A₀ and A₂ masked)",
            "units": [
                ("O₀", "obs", "clean"), ("A₀", "act", "masked"),
                ("O₁", "obs", "clean"), ("A₁", "act", "clean"),
                ("O₂", "obs", "clean"), ("A₂", "act", "masked"),
                ("O₃", "obs", "clean"),
            ],
        },
        {
            "title": "5. AOMT-Mixed\n(A₀ and O₂ masked)",
            "units": [
                ("O₀", "obs", "clean"), ("A₀", "act", "masked"),
                ("O₁", "obs", "clean"), ("A₁", "act", "clean"),
                ("O₂", "obs", "masked"), ("A₂", "act", "clean"),
                ("O₃", "obs", "clean"),
            ],
        },
    ]

    COLORS = {
        "prompt":   "#2166AC",   # dark blue — clean context
        "clean":    "#92C5DE",   # light blue — clean bidirectional context
        "response": "#D6604D",   # red — masked response
        "masked":   "#B2182B",   # dark red — masked target
        "obs_clean":"#92C5DE",
        "act_clean":"#4DAC26",   # green — clean act
    }

    def unit_color(utype, state):
        if state in ("response", "masked"):
            return COLORS["masked"]
        if state == "prompt":
            return COLORS["prompt"]
        # clean
        return COLORS["obs_clean"] if utype == "obs" else COLORS["act_clean"]

    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 8))
    fig.suptitle("AOMT Training Examples — Unit-Level Masking Across All Methods",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, method in zip(axes, methods):
        units = method["units"]
        ax.set_xlim(0, len(units))
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(method["title"], loc="left", fontsize=9, pad=2)

        for i, (label, utype, state) in enumerate(units):
            color = unit_color(utype, state)
            is_masked = state in ("response", "masked")
            hatch = "////" if is_masked else ""
            edgecolor = "white" if is_masked else "#333333"

            rect = mpatches.FancyBboxPatch(
                (i + 0.05, 0.1), 0.85, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=edgecolor,
                linewidth=1.5, hatch=hatch
            )
            ax.add_patch(rect)

            # Label
            display = f"{label}\n[MASK]" if is_masked else label
            ax.text(i + 0.475, 0.5, display,
                    ha="center", va="center", fontsize=8,
                    color="white" if (state in ("prompt", "response", "masked")) else "#1a1a1a",
                    fontweight="bold" if is_masked else "normal")

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["prompt"],   label="Prompt context (clean)"),
        mpatches.Patch(color=COLORS["obs_clean"],label="Obs context (clean, bidirectional)"),
        mpatches.Patch(color=COLORS["act_clean"],label="Act context (clean, bidirectional)"),
        mpatches.Patch(color=COLORS["masked"], hatch="////", label="Masked target (loss here)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved PNG to: {output_path}")
    plt.close()


# ---- Token length distribution (bonus) --------------------------------------

def visualize_length_distribution(tokenizer, png_path=None):
    try:
        import matplotlib.pyplot as plt
        from datasets import load_dataset
    except ImportError:
        return

    print("\n[Length distribution] Loading dataset...")
    dataset = load_dataset("agent-eto/eto-sft-trajectory", split="train")

    action_lens, obs_lens = [], []
    for ex in dataset:
        for turn in ex["conversations"]:
            toks = tokenizer.encode(turn["value"], add_special_tokens=False)
            if turn["from"] == "gpt":
                action_lens.append(len(toks))
            else:
                obs_lens.append(len(toks))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, lens, label, gen_len in [
        (axes[0], action_lens, "Action token lengths", 256),
        (axes[1], obs_lens,   "Observation token lengths", None),
    ]:
        ax.hist(lens, bins=50, edgecolor="white", color="#2166AC", alpha=0.85)
        ax.set_xlabel("Token count")
        ax.set_ylabel("Frequency")
        ax.set_title(label)
        if gen_len:
            ax.axvline(gen_len, color="red", linewidth=2, linestyle="--",
                       label=f"gen_length={gen_len}")
            ax.legend()
        p99 = np.percentile(lens, 99)
        ax.axvline(p99, color="orange", linewidth=1.5, linestyle=":",
                   label=f"p99={p99:.0f}")
        ax.legend()

    plt.suptitle("Token Length Distributions — eto-sft-trajectory", fontweight="bold")
    plt.tight_layout()
    if png_path:
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Length distribution saved to {png_path}")
    else:
        plt.show()
    plt.close()


# ---- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise AOMT training examples")
    parser.add_argument("--png_out",   type=str, default=None,
                        help="Save PNG to this path (requires matplotlib)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to LLaDA tokenizer for real tokenisation")
    parser.add_argument("--no_terminal", action="store_true",
                        help="Skip terminal visualisation")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            print(f"Loaded tokenizer from {args.tokenizer}")
            print(f"  MASK_TOKEN_ID: {tokenizer.mask_token_id}")
            print(f"  EOS_TOKEN_ID:  {tokenizer.eos_token_id}")
        except Exception as e:
            print(f"Could not load tokenizer: {e}")

    if not args.no_terminal:
        visualize_terminal()

    if args.png_out:
        visualize_png(args.png_out, tokenizer)

    if tokenizer:
        lens_path = args.png_out.replace(".png", "_lengths.png") if args.png_out else None
        visualize_length_distribution(tokenizer, lens_path)


if __name__ == "__main__":
    main()
