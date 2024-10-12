instruction_gen_prompt = """
Generate an instruction. This instruction should be a question that humans would be ask. It can be in imperative or interrogative form. We will use the instructions you generate to train models, so you must ensure that the instructions generated are of high quality and correct and also keep the instruction clear and concise.
You should:
1. Briefly explain why you generate this instruction.
2. Think about whether you need to add some input to this instruction so that it can be answered directly. (For example, for tasks that involve summarizing, you need to provide the paragraph to be summarized).
3. Return you output strictly following the format: 

Your generated instruction should strictly follow the following format:
<instruction><YOUR INSTRUCTION HERE> <YOUR_INPUT_HERE></instruction>

If there is no need to add inputs to answer the instruction, you can skip the <YOUR_INPUT_HERE> part. If you need to add inputs, just replace the <YOUR_INPUT_HERE> with the input. Now here are some examples of reference instructions, and please generate only one instruction.
"""
