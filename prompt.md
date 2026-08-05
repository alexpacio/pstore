create a simple and fast GUI called "pstore" in rust to help a developer to generate an LLM prompt, allowing him to ask questions about things while writing the prompt using the most famous coding agents. it should also automatically figure out the right difficulty and then the right model to ask, optimizing costs and prioritizing speed of getting outputs for hints.

it should use coding agents already installed on the system and automatically detect and categorize them. also, should route automatically if one is unable because not logged or not abe to use the selecteed model for any reason. this means that the program should rank the best models automatically. perhaps worth exploring regolo.ai's opensource "brick" model.

The program should have a column on the left that allows to explore and select the prompts and on the right show a big pane to see and write the prompt. mouse cursor should be working to move the mouse. Prompt is expected to be in markdown.

as i said, while writing, the user should be able to ask for hints to an llm: prompt should be based on a text selection of what dev is writing or on a brand new text input.

there should be possible at the end of the writing process to  select:
- a button to go running regolo on the local gpu and, if not available, to the CPU, to start the ranking process to see the best model to use according to the ones available on the coding agents
- the prompt shrinker process to reduce its size while keeping the prompt context accurate. it should act over the current opened prompt.
- send end directly the prompt to the selected coding agent by opening it in another window.

The program should be written in rust and work on every major desktop platform.
prompts should be versioned, should be saved as md files into the current folder where pstore is opened (changable with env or cli option). copy paste and rollbacks (ctrl-z) should be working.