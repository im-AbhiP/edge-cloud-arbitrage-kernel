"""
Main interactive entry point for the Edge-Cloud AI Arbitrage Kernel.

THIS IS THE FILE YOU RUN TO ACTUALLY USE THE SYSTEM.

It provides an interactive command-line interface where you:
1. Type a prompt (question, task, request)
2. Optionally configure task parameters (or use smart defaults)
3. Watch the council deliberate in real-time
4. Get the final answer

The council of 5 local LLMs will:
- Vote on whether your prompt needs cloud or can be handled locally
- Vote on which local model should answer (if local)
- Generate an answer
- Review the answer iteratively until consensus

HOW TO RUN:
    In PyCharm: Right-click this file -> "Run 'main'"
    In terminal: python main.py

    For fast mode (fewer models, faster):
        python main.py --fast

    For specific council size:
        python main.py --models 3
        python main.py --models 3 --iterations 2 --majority

WHAT IS input()?
    Python's built-in function that pauses the program, shows a
    message on screen, waits for the user to type something and
    press Enter, then returns whatever they typed as a string.

    Example:
        name = input("What is your name? ")
        # Program pauses here, user types "Abhishek", presses Enter
        # name now contains "Abhishek"

WHAT IS sys.argv?
    When you run "python main.py --fast", sys.argv contains:
    ["main.py", "--fast"]
    We use argparse to parse these command-line arguments into
    structured options.

WHAT IS argparse?
    Python's built-in library for parsing command-line arguments.
    It automatically generates help text, validates inputs, and
    converts strings to the right types. When you run
    "python main.py --help", argparse shows all available options.
"""

import argparse
import sys
import time
from datetime import datetime

from council.agents import LLMCouncil
from council.deliberation import DeliberationResult
from runtime.tasks import TaskType, DataSensitivity
from runtime.logging_utils import CallLogger


# =====================================================================
# CONFIGURATION
# =====================================================================
# These constants define the default behavior of the interactive CLI.
# You can override them with command-line arguments.
# =====================================================================

# Which models participate in the council by default
# Excludes gpt-oss:20b because it uses ~14GB RAM and causes heavy
# swapping on 16GB M1 Pro. Include it with --include-large flag.
DEFAULT_COUNCIL_MODELS = [
    "ollama/llama3.1-8b",
    "ollama/deepseek-r1-8b",
    "ollama/qwen3-8b",
    "ollama/deepseek-coder-6.7b",
]

# All 5 models including the large one
FULL_COUNCIL_MODELS = [
    "ollama/gpt-oss-20b",
    "ollama/llama3.1-8b",
    "ollama/deepseek-r1-8b",
    "ollama/qwen3-8b",
    "ollama/deepseek-coder-6.7b",
]

# Fast mode uses only 3 models for quicker responses
FAST_COUNCIL_MODELS = [
    "ollama/llama3.1-8b",
    "ollama/qwen3-8b",
    "ollama/deepseek-r1-8b",
]

# Map user-friendly task type names to TaskType enum values
# This lets users type "research" instead of "DEEP_RESEARCH"
TASK_TYPE_MAP = {
    "qa": TaskType.QUICK_QA,
    "question": TaskType.QUICK_QA,
    "summary": TaskType.SIMPLE_SUMMARY,
    "research": TaskType.DEEP_RESEARCH,
    "code": TaskType.CODE_REVIEW,
    "planning": TaskType.PLANNING,
    "creative": TaskType.CREATIVE_WRITING,
    "data": TaskType.DATA_ANALYSIS,
    "long": TaskType.LONG_SUMMARIZATION,
}

# Map user-friendly sensitivity names to DataSensitivity enum values
SENSITIVITY_MAP = {
    "low": DataSensitivity.LOW,
    "medium": DataSensitivity.MEDIUM,
    "high": DataSensitivity.HIGH,
}


# =====================================================================
# DISPLAY HELPERS
# =====================================================================
# These functions format the council's output for the terminal.
# They make the interactive experience readable and informative.
# =====================================================================

def print_banner():
    """Print the application banner when the program starts."""
    print()
    print("=" * 60)
    print("  Edge-Cloud AI Arbitrage Kernel")
    print("  Council of Local LLMs - Interactive Mode")
    print("=" * 60)
    print()
    print("  Type your prompt and the council will deliberate.")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'help' for commands.")
    print("  Type 'stats' to see session statistics.")
    print()


def print_help():
    """
    Print available commands and tips.

    WHY A HELP COMMAND?
    Users (including you in 3 months when you've forgotten the
    details) need a quick reference for what they can do.
    """
    print()
    print("-" * 60)
    print("COMMANDS:")
    print("-" * 60)
    print("  help              Show this help message")
    print("  quit / exit       Exit the program")
    print("  stats             Show session statistics")
    print("  config            Show current council configuration")
    print()
    print("TASK TYPE PREFIXES (optional, add before your prompt):")
    print("  /qa         Quick question (default)")
    print("  /research   Deep research or analysis")
    print("  /code       Code review or generation")
    print("  /summary    Summarization task")
    print("  /planning   Planning or strategy task")
    print("  /creative   Creative writing task")
    print("  /data       Data analysis task")
    print()
    print("SENSITIVITY FLAGS (optional, add before your prompt):")
    print("  /private    Force local processing (HIGH sensitivity)")
    print("  /sensitive  Same as /private")
    print()
    print("EXAMPLES:")
    print('  What is the capital of France?')
    print('  /research Analyze AMD vs NVIDIA for AI inference')
    print('  /code Review this Python function: def add(a,b): return a+b')
    print('  /private Analyze this salary data: John earns $150k')
    print('  /research /private Internal competitive analysis of AMD')
    print("-" * 60)
    print()


def print_config(council_models, max_iterations, approval_threshold):
    """Print the current council configuration."""
    print()
    print("-" * 60)
    print("CURRENT CONFIGURATION:")
    print("-" * 60)
    print(f"  Council members:     {len(council_models)}")
    for model in council_models:
        print(f"    - {model}")
    print(f"  Max iterations:      {max_iterations}")
    print(f"  Approval threshold:  {approval_threshold:.0%}")
    print("-" * 60)
    print()


def print_result_summary(result: DeliberationResult):
    """
    Print a formatted summary of the deliberation result.

    This is what the user sees after the council finishes.
    It shows the final answer prominently, with metadata below.
    """
    print()
    print("=" * 60)
    print("COUNCIL ANSWER")
    print("=" * 60)
    print()
    print(result.final_answer)
    print()
    print("-" * 60)
    print("DELIBERATION DETAILS:")
    print(f"  Routing:     {result.routing_decision.upper()}")
    print(f"  Model:       {result.selected_model}")
    print(f"  Iterations:  {result.iterations}")
    print(f"  Consensus:   {'Yes' if result.consensus_reached else 'No'}")
    print(f"  Total time:  {result.total_deliberation_ms / 1000:.1f}s")

    if result.cloud_tier and result.routing_decision == "cloud":
        print(f"  Cloud tier:  {result.cloud_tier}")

    # Show routing vote breakdown if votes exist
    if result.routing_votes:
        local_votes = sum(
            1 for v in result.routing_votes
            if v.vote == "local" and v.success
        )
        cloud_votes = sum(
            1 for v in result.routing_votes
            if v.vote == "cloud" and v.success
        )
        print(f"  Votes:       local={local_votes} cloud={cloud_votes}")

    print("-" * 60)
    print()


def print_session_stats(session_results):
    """
    Print aggregate statistics for the current session.

    Tracks how many prompts were answered, routing decisions,
    average time, and consensus rate.
    """
    if not session_results:
        print("\n  No prompts processed yet.\n")
        return

    total = len(session_results)
    local_count = sum(
        1 for r in session_results
        if r.routing_decision == "local"
    )
    cloud_count = sum(
        1 for r in session_results
        if r.routing_decision == "cloud"
    )
    consensus_count = sum(
        1 for r in session_results
        if r.consensus_reached
    )
    avg_time = (
        sum(r.total_deliberation_ms for r in session_results) / total
    )
    total_time = sum(
        r.total_deliberation_ms for r in session_results
    )

    print()
    print("-" * 60)
    print("SESSION STATISTICS:")
    print("-" * 60)
    print(f"  Prompts processed:   {total}")
    print(f"  Routed locally:      {local_count} ({local_count/total*100:.0f}%)")
    print(f"  Routed to cloud:     {cloud_count} ({cloud_count/total*100:.0f}%)")
    print(f"  Consensus reached:   {consensus_count}/{total} ({consensus_count/total*100:.0f}%)")
    print(f"  Avg deliberation:    {avg_time/1000:.1f}s")
    print(f"  Total session time:  {total_time/1000:.1f}s ({total_time/60000:.1f}min)")

    # Show which models were selected most often
    model_counts = {}
    for r in session_results:
        model = r.selected_model
        model_counts[model] = model_counts.get(model, 0) + 1

    print(f"  Models selected:")
    for model, count in sorted(
        model_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"    {model}: {count} time(s)")

    print("-" * 60)
    print()


# =====================================================================
# PROMPT PARSING
# =====================================================================
# These functions parse the user's input to extract task type,
# sensitivity flags, and the actual prompt text.
# =====================================================================

def parse_user_input(raw_input: str):
    """
    Parse user input for optional prefixes and extract the prompt.

    Users can type:
        "What is TCP?"
        "/research Analyze AMD vs NVIDIA"
        "/private Salary data analysis"
        "/research /private Internal competitive analysis"

    This function extracts:
        - task_type (from /qa, /research, /code, etc.)
        - data_sensitivity (from /private or /sensitive)
        - the actual prompt text (everything after prefixes)

    WHAT IS .startswith()?
        A string method that returns True if the string begins
        with the specified text. "hello".startswith("he") is True.

    WHAT IS .split()?
        Splits a string into a list by whitespace (by default).
        "hello world".split() gives ["hello", "world"].
    """
    task_type = TaskType.QUICK_QA
    data_sensitivity = DataSensitivity.LOW
    prompt = raw_input.strip()

    # Process prefix flags one at a time
    # We loop because there might be multiple prefixes:
    # "/research /private Some prompt"
    changed = True
    while changed:
        changed = False

        # Check for task type prefixes
        for prefix, ttype in TASK_TYPE_MAP.items():
            if prompt.lower().startswith(f"/{prefix}"):
                task_type = ttype
                # Remove the prefix from the prompt
                prompt = prompt[len(prefix) + 1:].strip()
                changed = True
                break

        # Check for sensitivity flags
        if prompt.lower().startswith("/private"):
            data_sensitivity = DataSensitivity.HIGH
            prompt = prompt[8:].strip()
            changed = True
        elif prompt.lower().startswith("/sensitive"):
            data_sensitivity = DataSensitivity.HIGH
            prompt = prompt[10:].strip()
            changed = True

    return prompt, task_type, data_sensitivity


def estimate_complexity(prompt: str, task_type: TaskType) -> float:
    """
    Estimate how complex a prompt is, on a 0.0 to 1.0 scale.

    This is a simple heuristic, not a perfect classifier.
    The council's vote is the real decision-maker — this just
    provides a reasonable default for the TaskMetadata.

    HEURISTICS USED:
    - Longer prompts tend to be more complex
    - Certain task types are inherently more complex
    - Keywords like "analyze", "compare", "evaluate" suggest complexity
    """
    # Base complexity from task type
    type_complexity = {
        TaskType.QUICK_QA: 0.2,
        TaskType.SIMPLE_SUMMARY: 0.3,
        TaskType.CREATIVE_WRITING: 0.4,
        TaskType.CODE_REVIEW: 0.5,
        TaskType.DATA_ANALYSIS: 0.6,
        TaskType.LONG_SUMMARIZATION: 0.6,
        TaskType.PLANNING: 0.7,
        TaskType.DEEP_RESEARCH: 0.8,
    }
    complexity = type_complexity.get(task_type, 0.5)

    # Adjust based on prompt length
    # Longer prompts usually ask for more
    word_count = len(prompt.split())
    if word_count > 100:
        complexity = min(complexity + 0.2, 1.0)
    elif word_count > 50:
        complexity = min(complexity + 0.1, 1.0)
    elif word_count < 10:
        complexity = max(complexity - 0.1, 0.0)

    # Adjust based on complexity keywords
    complex_keywords = [
        "analyze", "compare", "evaluate", "assess", "critique",
        "implications", "tradeoffs", "comprehensive", "detailed",
        "in-depth", "multi-step", "strategy", "architecture",
    ]
    prompt_lower = prompt.lower()
    keyword_count = sum(
        1 for word in complex_keywords
        if word in prompt_lower
    )
    if keyword_count >= 3:
        complexity = min(complexity + 0.2, 1.0)
    elif keyword_count >= 1:
        complexity = min(complexity + 0.1, 1.0)

    return round(complexity, 2)


def estimate_importance(prompt: str, task_type: TaskType) -> float:
    """
    Estimate how important getting a high-quality answer is.

    Similar heuristic approach to complexity estimation.
    Research and planning tasks are treated as higher importance
    than quick Q&A.
    """
    type_importance = {
        TaskType.QUICK_QA: 0.3,
        TaskType.SIMPLE_SUMMARY: 0.4,
        TaskType.CREATIVE_WRITING: 0.4,
        TaskType.CODE_REVIEW: 0.6,
        TaskType.DATA_ANALYSIS: 0.7,
        TaskType.LONG_SUMMARIZATION: 0.5,
        TaskType.PLANNING: 0.7,
        TaskType.DEEP_RESEARCH: 0.8,
    }
    return type_importance.get(task_type, 0.5)


# =====================================================================
# COMMAND-LINE ARGUMENT PARSING
# =====================================================================

def parse_arguments():
    """
    Parse command-line arguments.

    WHAT IS argparse?
    Python's standard library for handling command-line options.
    When you run "python main.py --fast", argparse turns "--fast"
    into a Python boolean that your code can check.

    It also auto-generates help text:
        python main.py --help
    """
    parser = argparse.ArgumentParser(
        description="Edge-Cloud AI Arbitrage Kernel - Interactive Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Default: 4 models, 3 iterations
  python main.py --fast             # Fast: 3 models, 2 iterations
  python main.py --include-large    # Include gpt-oss:20b (slow but strong)
  python main.py --models 3         # Use only 3 council models
  python main.py --iterations 2     # Max 2 review rounds
  python main.py --majority         # 60% approval instead of unanimous
        """,
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 3 small models, 2 iterations, majority approval",
    )

    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include gpt-oss:20b in the council (uses ~14GB RAM, slow)",
    )

    parser.add_argument(
        "--models",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Number of council models to use (1-5)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Maximum review iterations (default: 3, fast: 2)",
    )

    parser.add_argument(
        "--majority",
        action="store_true",
        help="Use majority approval (60%%) instead of unanimous",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom approval threshold (0.0 to 1.0)",
    )

    return parser.parse_args()


# =====================================================================
# MAIN INTERACTIVE LOOP
# =====================================================================

def main():
    """
    Main function — sets up the council and runs the interactive loop.

    WHAT IS AN INTERACTIVE LOOP?
    It's a pattern where the program:
    1. Shows a prompt (>>> )
    2. Waits for user input
    3. Processes the input
    4. Shows the result
    5. Goes back to step 1

    This repeats until the user types 'quit' or 'exit'.
    This is sometimes called a REPL (Read-Eval-Print Loop).
    """

    # --- Parse command-line arguments ---
    args = parse_arguments()

    # --- Determine council configuration based on arguments ---

    # Select which models to use
    if args.fast:
        council_models = FAST_COUNCIL_MODELS
        max_iterations = 2
        approval_threshold = 0.67
    elif args.include_large:
        council_models = FULL_COUNCIL_MODELS
        max_iterations = 3
        approval_threshold = 1.0
    else:
        council_models = DEFAULT_COUNCIL_MODELS
        max_iterations = 3
        approval_threshold = 1.0

    # Override with explicit --models count
    if args.models is not None:
        council_models = FULL_COUNCIL_MODELS[:args.models]

    # Override iterations if specified
    if args.iterations is not None:
        max_iterations = args.iterations

    # Override threshold if specified
    if args.majority:
        approval_threshold = 0.6
    if args.threshold is not None:
        approval_threshold = args.threshold

    # --- Create the council ---
    print_banner()
    print_config(council_models, max_iterations, approval_threshold)

    council = LLMCouncil(
        council_models=council_models,
        max_iterations=max_iterations,
        approval_threshold=approval_threshold,
    )

    # Track session results for statistics
    session_results = []
    logger = CallLogger()

    # --- Interactive loop ---
    # This is the main loop that keeps the program running
    # until the user types 'quit' or 'exit'.
    while True:
        try:
            # Show the prompt and wait for user input
            # input() blocks (pauses) until the user presses Enter
            raw_input_text = input("\n>>> ").strip()

            # --- Handle empty input ---
            if not raw_input_text:
                continue

            # --- Handle commands ---
            # .lower() converts to lowercase so "QUIT" works too
            command = raw_input_text.lower()

            if command in ("quit", "exit", "q"):
                print("\nExiting...")
                if session_results:
                    print_session_stats(session_results)
                print("Goodbye! Check data/logs.db for full audit trail.")
                print("Run: streamlit run dashboard/streamlit_app.py")
                break

            if command == "help":
                print_help()
                continue

            if command == "stats":
                print_session_stats(session_results)
                continue

            if command == "config":
                print_config(
                    council_models, max_iterations, approval_threshold
                )
                continue

            # --- Parse the prompt for prefixes ---
            prompt, task_type, data_sensitivity = parse_user_input(
                raw_input_text
            )

            # Validate that we have an actual prompt after parsing
            if not prompt:
                print("  Please enter a prompt after the command prefix.")
                continue

            # --- Estimate complexity and importance ---
            complexity = estimate_complexity(prompt, task_type)
            importance = estimate_importance(prompt, task_type)

            # --- Show what we parsed ---
            print()
            print(f"  Task type:    {task_type.value}")
            print(f"  Sensitivity:  {data_sensitivity.value}")
            print(f"  Complexity:   {complexity}")
            print(f"  Importance:   {importance}")

            if data_sensitivity == DataSensitivity.HIGH:
                print("  ** PRIVACY: This will be processed locally only **")

            # --- Run the council deliberation ---
            result = council.ask(
                prompt=prompt,
                task_type=task_type,
                complexity=complexity,
                importance=importance,
                data_sensitivity=data_sensitivity,
                budget_sensitivity=0.5,
            )

            # --- Display the result ---
            print_result_summary(result)

            # --- Track for session statistics ---
            session_results.append(result)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            # KeyboardInterrupt is raised when the user presses Ctrl+C.
            # Without this handler, the program would crash with a
            # traceback. With it, we exit cleanly.
            print("\n\nInterrupted by user.")
            if session_results:
                print_session_stats(session_results)
            print("Goodbye!")
            break

        except Exception as e:
            # Catch any unexpected errors and continue rather than crash.
            # This is important for an interactive tool — one bad prompt
            # shouldn't kill the entire session.
            print(f"\n  ERROR: {e}")
            print("  The council encountered an error. Try again.")
            print("  If this persists, check that Ollama is running:")
            print("    ollama serve")
            continue


# =====================================================================
# ENTRY POINT
# =====================================================================
# WHAT IS if __name__ == "__main__"?
#
# When you run "python main.py", Python sets a special variable
# called __name__ to "__main__" for that file. This check means:
# "Only run main() if this file was executed directly."
#
# If someone imports this file (from main import parse_user_input),
# main() does NOT run automatically. This prevents the interactive
# loop from starting when you just want to use a function from
# this file.
#
# This is a Python best practice called the "name-main idiom".
# =====================================================================

if __name__ == "__main__":
    main()
