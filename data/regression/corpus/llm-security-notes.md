# Internal Notes on LLM Application Security

These notes summarise how the platform team reasons about language-model risk.
They are included in the evaluation corpus deliberately: a document that
*discusses* prompt injection must not be mistaken for a document that *attempts*
it. A detector that cannot tell the difference is unusable on a security corpus.

## Indirect prompt injection

Indirect prompt injection is the case where hostile text reaches the model
through retrieved content rather than through the user's message. The classic
example instructs the assistant to disregard its configuration and to send
retrieved material to an external endpoint.

The mitigation the team relies on is structural rather than lexical. Retrieved
passages are placed in the user role inside delimiters that the retrieved text
could not have predicted, and the system message states that evidence blocks are
data. Detection is a second layer, not the first.

## Why keyword filtering is insufficient

A filter that matches the literal phrase "ignore previous instructions" is
defeated by homoglyph substitution, by zero-width characters inserted between
letters, and by paraphrase. Normalisation before matching closes the first two.
Nothing closes the third completely.

## Signal quality

Reviewers should ignore the noise floor of low-severity findings when triaging.
A single finding on a security paper is almost always a false positive; a
majority of retrieved passages carrying findings is a compromised corpus.

## Residual risk

The team's position is that prompt injection is an unsolved problem and that any
claim of complete protection is false. The controls that do not depend on
detection are the ones that carry the weight: the assistant exposes no tool that
a model could be persuaded to invoke, and no answer is returned unless its
claims trace back to retrieved evidence.
