#!/usr/bin/env python3
"""
visualizechart.py

This script reads normalized CSV data and trade events from files provided as arguments,
and generates a PineScript v5 (.ps5) file that you can paste into TradingView to visualize
the normalized data and the trade events (entries/exits) recorded by a worker.

Usage example:
    python visualizechart.py --iter 368 --worker tf2_w3 --data /path/to/normalized_data.csv --events /path/to/trade_events.json
"""

import argparse
import json
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generate PineScript v5 file from normalized data and trade events")
    parser.add_argument("--iter", type=int, required=True, help="Current iteration number")
    parser.add_argument("--worker", type=str, required=True, help="Worker ID")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file containing normalized data")
    parser.add_argument("--events", type=str, required=True, help="Path to JSON file containing trade events")
    args = parser.parse_args()

    # Read CSV data (assumes CSV has header and one of the columns is a close price, e.g., "close_2")
    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Look for a column starting with "close_"
    close_columns = [col for col in df.columns if col.lower().startswith("close_")]
    if not close_columns:
        print("No close column found in CSV data.")
        return
    close_col = close_columns[0]
    normalized_close = df[close_col].tolist()

    # Read trade events from JSON
    try:
        with open(args.events, "r") as f:
            trade_events = json.load(f)
    except Exception as e:
        print(f"Error reading trade events JSON file: {e}")
        return

    # Build a PineScript array string from the normalized close data
    data_array_str = ", ".join(f"{x}" for x in normalized_close)

    # For each trade event, build a PineScript snippet to plot a label.
    # Each event is expected to have: "type" ("open" or "close"), "timestamp", "price", and "position"
    event_lines = []
    for event in trade_events:
        try:
            ts = int(event.get("timestamp", 0))
            price = event.get("price", 0)
            event_type = event.get("type", "unknown")
            # Label text and color based on event type and position:
            if event_type == "open":
                label_text = "Open " + ("Long" if event.get("position", 1) == 1 else "Short")
                color = "green" if event.get("position", 1) == 1 else "red"
            elif event_type == "close":
                label_text = "Close"
                color = "blue"
            else:
                label_text = event_type
                color = "gray"
            # Generate an if-statement in PineScript that creates a label when the bar time equals the event timestamp.
            # (Assumes the CSV timestamps are in Unix time.)
            event_line = f"""if (time == {ts})
    label.new(bar_index, {price}, "{label_text}", style=label.style_label_up, color=color.{color})"""
            event_lines.append(event_line)
        except Exception as e:
            print(f"Error processing an event: {e}")

    # Build the complete PineScript code.
    pinescript_code = f"""//@version=5
indicator("Worker {args.worker} - Iteration {args.iter}", overlay=true)

// Normalized close data array
var float[] closeData = array.from({data_array_str})

// Plot the normalized close data.
// (Since the number of bars in the imported array may be less than the number of bars on the chart,
//  we use modulo indexing to loop the data.)
plot(array.get(closeData, bar_index % array.size(closeData)), title="Normalized Close", color=color.blue, linewidth=2)

// Trade events markers
{"\n".join(event_lines)}
"""

    # Save the PineScript file (.ps5) in the same directory as the CSV file.
    output_folder = os.path.dirname(args.data)
    output_filename = f"chart_worker_{args.worker}_iter_{args.iter}.ps5"
    output_path = os.path.join(output_folder, output_filename)
    try:
        with open(output_path, "w") as f:
            f.write(pinescript_code)
        print(f"PineScript file saved to: {output_path}")
    except Exception as e:
        print(f"Error writing PineScript file: {e}")

if __name__ == "__main__":
    main()
