# Temporal Information Model

## Primary Timeline

The primary decision timeline is M15.

Each prediction instance is associated with an M15 decision timestamp.

## Multi-Timeframe Information

### M15
M15 provides the primary short-term market dynamics.

Only information observable at or before the decision point may be used.

### H1
H1 provides intermediate-term market context.

Two categories are considered:

1. Completed H1 context
2. Current H1 partial state reconstructed only from information available up to the decision time

Final values of an unfinished H1 candle must not be used.

### D1
D1 provides long-term market context and market-regime information.

Two categories are considered:

1. Completed D1 context
2. Current D1 partial state reconstructed only from information available up to the decision time

Final values of an unfinished D1 candle must not be used.

## Causality Principle

Every feature must be computable using only information that was observable at the decision timestamp.

Future market information must never be used for feature construction.

## Target Separation

Future information may be used only for target/label construction according to the predefined prediction horizon.

Future information must never enter the feature set.

## Research Scope

The framework preserves the multi-timeframe structure:

M15 + H1 + D1

and does not discard higher-timeframe information merely because it is not fully completed.

The usefulness of completed and partial higher-timeframe information will be evaluated experimentally and through ablation studies.