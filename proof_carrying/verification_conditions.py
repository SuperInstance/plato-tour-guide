#!/usr/bin/env python3
"""
Verification Condition Generator for Plato Tour Guide

Converts theorem specifications into SMT-LIB verification conditions
that can be checked by Z3, CVC5, or other SMT solvers.

Usage:
    python verification_conditions.py --theorem h1_cohomology --output vc.smt2
"""

import argparse
import textwrap
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
from enum import Enum


class VCType(Enum):
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    SAFETY = "safety"
    LIVENESS = "liveness"


@dataclass
class VerificationCondition:
    """A single verification condition"""
    name: str
    vc_type: VCType
    formula: str
    location: str  # Function/loop location
    comment: str


class VCGenerator:
    """Generate verification conditions from theorem specs"""

    def __init__(self, theorem: str):
        self.theorem = theorem
        self.vcs: List[VerificationCondition] = []
        self.declarations: Set[str] = set()
        self.assumptions: List[str] = []

    def add_declaration(self, decl: str):
        """Add a type/function declaration"""
        self.declarations.add(decl)

    def add_assumption(self, assumption: str):
        """Add a global assumption (axiom)"""
        self.assumptions.append(assumption)

    def generate_h1_cohomology_vcs(self):
        """Generate VCs for H1 Cohomology computation"""

        # Type declarations
        self.add_declaration("(declare-sort Graph)")
        self.add_declaration("(declare-sort Vertex)")
        self.add_declaration("(declare-sort Edge)")
        self.add_declaration("(declare-fun vertices (Graph) Int)")
        self.add_declaration("(declare-fun edges (Graph) Int)")
        self.add_declaration("(declare-fun is_flexible (Graph) Bool)")
        self.add_declaration("(declare-fun beta1 (Graph) Int)")

        # Preconditions
        self.vcs.append(VerificationCondition(
            name="h1_pre_valid_graph",
            vc_type=VCType.PRECONDITION,
            location="h1_cohomology:entry",
            comment="Input graph must be well-formed",
            formula="(=> (call h1_cohomology g) (and (>= (vertices g) 0) (>= (edges g) 0)))"
        ))

        self.vcs.append(VerificationCondition(
            name="h1_pre_connected",
            vc_type=VCType.PRECONDITION,
            location="h1_cohomology:entry",
            comment="Graph must be weakly connected for Betti number formula",
            formula="(=> (call h1_cohomology g) (is_connected g))"
        ))

        # Postconditions
        self.vcs.append(VerificationCondition(
            name="h1_post_betti_formula",
            vc_type=VCType.POSTCONDITION,
            location="h1_cohomology:exit",
            comment="Betti number formula: β₁ = |E| - |V| + 1",
            formula="(let ((beta (beta1 g)) (v (vertices g)) (e (edges g))) (=> (call h1_cohomology g) (= beta (- (- e v) 1))))"
        ))

        self.vcs.append(VerificationCondition(
            name="h1_post_flexibility",
            vc_type=VCType.POSTCONDITION,
            location="h1_cohomology:exit",
            comment="β₁ > 0 iff graph is flexible",
            formula="(let ((beta (beta1 g))) (=> (call h1_cohomology g) (= (> beta 0) (is_flexible g))))"
        ))

        # Loop invariants (for union-find iteration)
        self.vcs.append(VerificationCondition(
            name="h1_loop_invariant_components",
            vc_type=VCType.INVARIANT,
            location="union_find:while_edges",
            comment="Number of components = |V| - edges_processed + merges",
            formula="""
                (let ((c (count_components uf)) (v (vertices g)) (e_proc (edges_processed)) (m (merges)))
                  (and (>= c 1) (<= c v) (= c (- (- v e_proc) m))))
            """
        ))

        # Safety conditions
        self.vcs.append(VerificationCondition(
            name="h1_safety_no_overflow",
            vc_type=VCType.SAFETY,
            location="h1_cohomology:array_access",
            comment="Array access must be in bounds",
            formula="(forall ((i Int)) (=> (and (call h1_cohomology g) (access_parent i)) (and (>= i 0) (< i (vertices g)))))"
        ))

        # Complexity bounds (as arithmetic constraints)
        self.vcs.append(VerificationCondition(
            name="h1_complexity_time",
            vc_type=VCType.SAFETY,
            location="h1_cohomology:exit",
            comment="Time complexity O(|E|)",
            formula="(let ((e (edges g)) (t (execution_time))) (=> (call h1_cohomology g) (<= t (* c e))))"
        ))

    def generate_zhc_consensus_vcs(self):
        """Generate VCs for ZHC Consensus"""

        # Type declarations
        self.add_declaration("(declare-sort Agent)")
        self.add_declaration("(declare-sort Value)")
        self.add_declaration("(declare-fun num_agents () Int)")
        self.add_declaration("(declare-fun num_byzantine () Int)")
        self.add_declaration("(declare-fun is_honest (Agent) Bool)")
        self.add_declaration("(declare-fun decision (Agent) Value)")
        self.add_declaration("(declare-fun consensus_value () Value)")

        # Preconditions
        self.vcs.append(VerificationCondition(
            name="zhc_pre_byzantine_bound",
            vc_type=VCType.PRECONDITION,
            location="zhc_consensus:entry",
            comment="Byzantine agents must be less than 1/3 of total",
            formula="(< (* 3 (num_byzantine)) (num_agents))"
        ))

        self.vcs.append(VerificationCondition(
            name="zhc_pre_sufficient_honest",
            vc_type=VCType.PRECONDITION,
            location="zhc_consensus:entry",
            comment="At least 2f+1 honest agents required",
            formula="(>= (count_honest) (+ (* 2 (num_byzantine)) 1))"
        ))

        # Postconditions
        self.vcs.append(VerificationCondition(
            name="zhc_post_agreement",
            vc_type=VCType.POSTCONDITION,
            location="zhc_consensus:exit",
            comment="All honest agents decide same value",
            formula="""
                (forall ((a1 Agent) (a2 Agent))
                  (=> (and (is_honest a1) (is_honest a2))
                      (= (decision a1) (decision a2))))
            """
        ))

        self.vcs.append(VerificationCondition(
            name="zhc_post_validity",
            vc_type=VCType.POSTCONDITION,
            location="zhc_consensus:exit",
            comment="If leader honest, consensus equals leader's input",
            formula="(=> (is_honest leader) (= (consensus_value) (leader_input)))"
        ))

        # Safety conditions
        self.vcs.append(VerificationCondition(
            name="zhc_safety_byzantine_detection",
            vc_type=VCType.SAFETY,
            location="zhc_consensus:check_threshold",
            comment="Return error if Byzantine bound violated",
            formula="""
                (=> (>= (* 3 (num_byzantine)) (num_agents))
                    (is_error (zhc_consensus)))
            """
        ))

        # Timing constraints
        self.vcs.append(VerificationCondition(
            name="zhc_latency_bound",
            vc_type=VCType.POSTCONDITION,
            location="zhc_consensus:exit",
            comment="Decision must be reached within 38ms",
            formula="(let ((t (time_to_decision))) (=> (terminated_successfully) (<= t 38)))"
        ))

    def generate_pythagorean48_vcs(self):
        """Generate VCs for Pythagorean48 encoding"""

        # Type declarations
        self.add_declaration("(declare-sort Vector)")
        self.add_declaration("(declare-sort Code)")
        self.add_declaration("(declare-fun dimension () Int)")
        self.add_declaration("(declare-fun codebook_size () Int)")
        self.add_declaration("(declare-fun encode (Vector) Code)")
        self.add_declaration("(declare-fun decode (Code) Vector)")
        self.add_declaration("(declare-fun dist_sq (Vector Vector) Real)")
        self.add_declaration("(declare-fun in_unit_sphere (Vector) Bool)")

        # Preconditions
        self.vcs.append(VerificationCondition(
            name="py48_pre_dimension",
            vc_type=VCType.PRECONDITION,
            location="pythagorean48:encode",
            comment="Codebook dimension must be at least 48",
            formula="(>= (dimension) 48)"
        ))

        self.vcs.append(VerificationCondition(
            name="py48_pre_codebook_size",
            vc_type=VCType.PRECONDITION,
            location="pythagorean48:encode",
            comment="Codebook must contain exactly 48 vectors",
            formula "(= (codebook_size) 48)"
        ))

        # Postconditions
        self.vcs.append(VerificationCondition(
            name="py48_post_roundtrip",
            vc_type=VCType.POSTCONDITION,
            location="pythagorean48:decode",
            comment="Decoding encoded vector returns approximation",
            formula="""
                (forall ((x Vector))
                  (let ((enc (encode x)) (dec (decode enc)))
                    (=> (in_unit_sphere x)
                        (<= (dist_sq x dec) epsilon))))
            """
        ))

        self.vcs.append(VerificationCondition(
            name="py48_post_efficiency",
            vc_type=VCType.POSTCONDITION,
            location="pythagorean48:encode",
            comment="Bits per vector = log2(48) ≈ 5.585",
            formula "(= (bits_per_vector) 5.585)"
        ))

        # Safety conditions
        self.vcs.append(VerificationCondition(
            name="py48_safety_injective",
            vc_type=VCType.SAFETY,
            location="pythagorean48:encode",
            comment="Encoding is injective on well-separated inputs",
            formula="""
                (forall ((x Vector) (y Vector))
                  (=> (and (in_unit_sphere x) (in_unit_sphere y)
                           (> (dist_sq x y) epsilon))
                      (not (= (encode x) (encode y)))))
            """
        ))

        # Optimality conditions
        self.vcs.append(VerificationCondition(
            name="py48_optimal_separation",
            vc_type=VCType.POSTCONDITION,
            location="pythagorean48:codebook_check",
            comment="Codebook vectors are maximally separated",
            formula="""
                (forall ((i Int) (j Int))
                  (=> (and (>= i 0) (< i 48) (>= j 0) (< j 48) (not (= i j)))
                      (>= (dist_sq (nth_codebook i) (nth_codebook j)) min_separation)))
            """
        ))

    def to_smtlib(self) -> str:
        """Convert all VCs to SMT-LIB format"""

        output = []

        # Header
        output.append("; Auto-generated verification conditions")
        output.append(f"; Theorem: {self.theorem}")
        output.append("(set-logic ALL)")
        output.append("")

        # Declarations
        output.append("; Type and function declarations")
        for decl in sorted(self.declarations):
            output.append(decl)
        output.append("")

        # Assumptions/Axioms
        if self.assumptions:
            output.append("; Global assumptions")
            for assumption in self.assumptions:
                output.append(f"(assert {assumption})")
            output.append("")

        # Verification conditions
        output.append("; Verification conditions")
        for vc in self.vcs:
            output.append(f"; {vc.comment}")
            output.append(f"; Location: {vc.location}")
            output.append(f"(define-fun vc-{vc.name} () Bool")
            output.append(f"  {vc.formula})")
            output.append(f"(assert (vc-{vc.name}))")
            output.append("")

        # Check sat
        output.append("; Check satisfiability")
        output.append("(check-sat)")
        output.append("(get-model)")

        return "\n".join(output)

    def to_coq(self) -> str:
        """Convert VCs to Coq specification"""

        output = []
        output.append(f"(* Verification conditions for {self.theorem} *)")
        output.append("")

        for vc in self.vcs:
            output.append(f"(* {vc.comment} *)")
            output.append(f"(* Location: {vc.location} *)")

            # Convert SMT formula to Coq-like syntax
            coq_formula = self.smt_to_coq(vc.formula)
            output.append(f"Definition vc_{vc.name} : Prop := {coq_formula}.")
            output.append("")

        return "\n".join(output)

    def smt_to_coq(self, formula: str) -> str:
        """Simple SMT-LIB to Coq syntax converter"""
        # This is a simplified conversion
        formula = formula.replace("=>", "->")
        formula = formula.replace("forall", "forall")
        formula = formula.replace("and", "/\\")
        formula = formula.replace("or", "\\/")
        return formula


def main():
    parser = argparse.ArgumentParser(description="Generate verification conditions")
    parser.add_argument("--theorem", choices=["h1_cohomology", "zhc_consensus", "pythagorean48"],
                        required=True, help="Theorem to generate VCs for")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["smt2", "coq"], default="smt2",
                        help="Output format")

    args = parser.parse_args()

    generator = VCGenerator(args.theorem)

    # Generate VCs based on theorem
    if args.theorem == "h1_cohomology":
        generator.generate_h1_cohomology_vcs()
    elif args.theorem == "zhc_consensus":
        generator.generate_zhc_consensus_vcs()
    elif args.theorem == "pythagorean48":
        generator.generate_pythagorean48_vcs()

    # Write output
    if args.format == "smt2":
        content = generator.to_smtlib()
    else:
        content = generator.to_coq()

    with open(args.output, 'w') as f:
        f.write(content)

    print(f"Generated {len(generator.vcs)} verification conditions")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
