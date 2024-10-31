import anthropic

class Claude:

    def __init__(self,
                 api_key: str = "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 "):
        self.api_key = api_key

    def call_claude(self,
                    propmt: str,
                    max_tokens: int = 4000,
                    model: str = 'claude-2'):
        
        client = anthropic.Anthropic(api_key=self.api_key)
        scratchpad_blocks = client.messages.create(model="claude-3-opus-20240229",
                                                   max_tokens=max_tokens,
                                                   temperature=0,
                                                   messages=[{
                                                       "role": "user",
                                                       "content": [{
                                                           "type": "text",
                                                           "text": propmt
                                                       }]
                                                   }]).content
        scratchpad_block = scratchpad_blocks[0]
        scratchpad_str = scratchpad_block.text 
        return scratchpad_str       