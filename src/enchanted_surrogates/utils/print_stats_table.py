import shutil
from datetime import datetime


def print_stats_table(stats):
    """
    Prints a formatted, bordered, and centered table of run statistics in the terminal.

    This function takes a dictionary containing metadata and statistics for a run (such as a game session,
    simulation, or process) and displays it in a visually appealing format. It includes a bold header,
    optional subheader, and a bordered list of key-value pairs, all centered within the terminal window.

    Parameters:
    -----------
    stats : dict
        A dictionary containing run metadata and statistics. It should include:
        - "header" (str): The main title of the summary (displayed in bold uppercase).
        - "subheader" (str, optional): A subtitle or contextual note (e.g., date, run ID).
        - Other key-value pairs representing statistics (e.g., "Duration": "3h 14m").

    Example:
    --------
    run_stats = {
        "header": "Enchanted Surrogates",
        "subheader": "Run #42 - October 2, 2025",
        "Duration": "3h 14m",
        "Enemies Defeated": 128,
        "Spells Cast": 47,
        "Artifacts Found": 5,
        "Final Score": 9820
    }

    print_stats_table(run_stats)

    Output:
    -------
    A centered, bordered block in the terminal that looks like:

    +---------------------------------------------+
    |           ** ENCHANTED SURROGATES **        |
    |         -- Run #42 - October 2, 2025 --     |
    +---------------------------------------------+
    |  Duration            : 3h 14m               |
    |  Enemies Defeated    : 128                  |
    |  Spells Cast         : 47                   |
    |  Artifacts Found     : 5                    |
    |  Final Score         : 9820                 |
    +---------------------------------------------+

    Notes:
    ------
    - The output adapts to the current terminal width.
    - All content is padded and aligned for readability.
    - The function uses `shutil.get_terminal_size()` to determine centering.
    """
    # Extract header and subheader
    header = stats.get("header", "STATS").upper()
    subheader = stats.get("subheader", "")

    # Remove header and subheader from stats
    stats = {k: v for k, v in stats.items() if k not in ["header", "subheader"]}

    # Get terminal width
    term_width = shutil.get_terminal_size().columns

    # Format header and subheader
    header_line = f"** {header} **"
    subheader_line = f"-- {subheader} --" if subheader else ""

    # Format stats
    stat_lines = [f"{key:<20}: {value}" for key, value in stats.items()]
    max_stat_width = max(len(line) for line in stat_lines)
    box_width = max(len(header_line), len(subheader_line), max_stat_width) + 6

    # Build border
    border = "+" + "-" * (box_width - 2) + "+"

    # Center lines
    def center(line): return line.center(box_width)

    # Assemble output
    output = [border]
    output.append(center(header_line))
    if subheader_line:
        output.append(center(subheader_line))
    output.append(center(f'{datetime.now()}'))
    output.append(border)
    for line in stat_lines:
        output.append(f"|  {line.ljust(box_width - 4)}  |")
    output.append(border)

    # Print centered in terminal
    for line in output:
        print(line.center(term_width))
