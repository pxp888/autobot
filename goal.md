# autobot

there is a kev server running at 192.168.0.189:8009

I'd like to use it as the basis of an inference routing system.  

The goal is to develop a fastapi instance that pretends to be an inference provider, we'll call this autobot.  
autobot has a list of actual model provider details, including a description of what each model/provider is suited to.  

keep in mind the kev server does not need to see the full prompt, (i think it only looks at the first 2000 tokens anyway).  

when an inference request arrives at autobot, it should use the kev server to determine which model is best suited, and route the request to that provider.  
This should support streamintg requests as well as non-streaming.  

We can start with file based information for routing, but a webUI wouldn't hurt.  




for example: 


 {  "providers": {
    "thinkfarm": {
      "baseUrl": "https://app.thinkfarm.eu/v1",
      "api": "openai-completions",
      "apiKey": "8b4d68ce-82f6-4583-8718-7e21a0703f43",
      "models": [
        { "id": "qwen3.8:27b-ud-q4_k_m" },
        { "id": "qwen3.6:35b-a3b-ud-q4_k_m" }
      ]
    }
  }
}

qwen3.8:27b is the smart model, used for all advanced code generation.  
qwen3.6:35b is the fast worker.  minor edits, repo-search, easy work.  


