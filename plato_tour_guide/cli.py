#!/usr/bin/env python3
"""
plato-tour-guide CLI — dead simple terminal interface.

Usage:
    python -m plato_tour_guide.cli ask "how does H1 work?"
    python -m plato_tour_guide.cli merge "H1 is the first" "H1 is homology" "H1 is Betti"
    python -m plato_tour_guide.cli status
    python -m plato_tour_guide.cli help
    python -m plato_tour_guide.cli find "pythagorean"
"""

import sys
import argparse

from plato_tour_guide import TourGuideAgent, Tile


def cmd_ask(args):
    guide = TourGuideAgent(room_name=args.room or "fleet-math")
    answer, level, confidence = guide.handle(args.question, user_mode=args.mode)
    print(f"\n[Level {level} | confidence {confidence:.2f}]")
    print(answer)
    if args.verbose:
        print(f"\nFull context:")
        print(f"  Room: {guide.room_name}")
        print(f"  Mode: {args.mode}")
        print(f"  Cascade level: {level}")


def cmd_merge(args):
    from plato_tour_guide.consensus import consensus_snap, PartialAnswer
    
    partials = [
        PartialAnswer(
            room=f"agent-{i}", 
            answer=a, 
            confidence=0.7,
            reasoning="merge from CLI"
        )
        for i, a in enumerate(args.answers)
    ]
    
    tile = consensus_snap(partials, " ".join(args.answers))
    
    if tile:
        print(f"\n[confidence {tile.confidence:.2f} | {tile.partials_count} partials | spread {tile.spread:.3f}]")
        print(f"Type: {tile.notes}")
        print(f"\nConsensus answer:\n{tile.answer}")
    else:
        print("\nNo consensus — partials too different.")
        print("Consider adding more context or breaking into simpler questions.")


def cmd_status(args):
    guide = TourGuideAgent(room_name=args.room or "fleet-math")
    status = guide.get_status()
    
    print(f"\nTour Guide: {status['room']}")
    print(f"  Total tiles: {status['total_tiles']}")
    print(f"  Orientation tiles: {status['orientation_tiles']}")
    print(f"  Direct tiles: {status['direct_tiles']}")
    print(f"  Neighbors: {', '.join(status['neighbors']) or '(none configured)'}")
    print(f"  Cache age: {status['cache_age_seconds']:.1f}s")


def cmd_find(args):
    guide = TourGuideAgent(room_name=args.room or "fleet-math")
    tiles = guide._get_tiles()
    
    matches = []
    for tile in tiles:
        if tile.is_orientation():
            continue
        q_lower = tile.question.lower()
        if any(kw.lower() in q_lower for kw in args.keyword):
            matches.append(tile)
    
    if not matches:
        print(f"No tiles matching: {args.keyword}")
        return
    
    print(f"\nFound {len(matches)} tiles matching: {args.keyword}")
    for t in matches[:10]:
        print(f"\n  Q: {t.question[:60]}...")
        print(f"  A: {t.answer[:80]}...")
        print(f"  conf={t.confidence:.2f} | tags={t.tags[:3]}")


def cmd_help(args):
    print("""
PLATO Tour Guide — Wayfinding Science as Technology

QUICK START:
  plato-tour-guide ask "how does H1 work?"
  plato-tour-guide merge "H1 is the first" "H1 is homology"
  plato-tour-guide status
  plato-tour-guide find pythagorean

MODES:
  --mode morning  = Fresh developer (orientation first)
  --mode afternoon = Post-experience (diagnosis mode)
  --mode unknown   = Auto-detect (default)

EXAMPLES:
  $ plato-tour-guide ask "what is Laman's theorem" --mode morning
  $ plato-tour-guide merge "consensus via H1" "H1 detects holes" "use cohomology"
  $ plato-tour-guide status --room fleet-math
  $ plato-tour-guide find h1 homology

MORE:
  python -m plato_tour_guide.cli --help
""")


def main():
    parser = argparse.ArgumentParser(
        description="PLATO Tour Guide CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", help="Command")
    
    # ask
    ask_p = subparsers.add_parser("ask", help="Ask a question")
    ask_p.add_argument("question", help="The question to ask")
    ask_p.add_argument("--room", "-r", default="fleet-math", help="Room name")
    ask_p.add_argument("--mode", "-m", default="unknown", 
                       choices=["morning", "afternoon", "unknown"],
                       help="User mode")
    ask_p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # merge
    merge_p = subparsers.add_parser("merge", help="Merge partial answers")
    merge_p.add_argument("answers", nargs="+", help="Partial answers to merge")
    
    # status
    status_p = subparsers.add_parser("status", help="Show room status")
    status_p.add_argument("--room", "-r", default="fleet-math", help="Room name")
    
    # find
    find_p = subparsers.add_parser("find", help="Find tiles by keyword")
    find_p.add_argument("keyword", nargs="+", help="Keywords to search")
    find_p.add_argument("--room", "-r", default="fleet-math", help="Room name")
    
    # help
    subparsers.add_parser("help", help="Show help")
    
    args = parser.parse_args()
    
    if args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "merge":
        cmd_merge(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "find":
        cmd_find(args)
    else:
        cmd_help(args)


if __name__ == "__main__":
    main()