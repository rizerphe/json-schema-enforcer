# JSON schema enforcer

A progressive JSON schema validator. For use with large language models, to force their output to strictly adhere to a pre-defined schema. The goal is to be able to validate whether the partial JSON output breaks the schema, in a way where you can't get yourself into a dead end - if "...ab" is valid, then there is an x for which "...abx" is valid, all the way until you got the complete output.

This is a quick experiment to show how easy it is to implement a JSON schema enforcer. I made it out of spite after OpenAI didn't enforce the schema of the returns of the function call feature, and I made it in a day, which does prove my point - OpenAI *could* have done this, they just didn't bother.

Note that I am not a programming language developer, not at all, therefore I fully expect this code to be _very shitty._

Another note: I am not using the full JSON schema specification, but my own simplified standard similar to the one OpenAI use.
