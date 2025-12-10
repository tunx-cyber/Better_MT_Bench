基于fastcht/llm_judge的更加灵活的MT-Bench。
- 支持最新的openai API。不在会被fastchat老版本的openai与vllm不兼容的问题所折磨。
- 支持各种模型，只需要根据函数的参数要求，可以自行定义judge model。
- 更快的生成速度。使用vllm帮助你的模型更快生成，并且支持lora参数的插入。
- 原汁原味的MT-Bench，prompt完全根据fastcht/llm_judge进行生成。
