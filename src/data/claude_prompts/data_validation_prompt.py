source_code_validation_prompt = """

The aim of this task is to refine the source code given in below 

<code>
{source_code}
</code> 

in response to the error message

<error_msg>
{error_message}
</error_msg>

In particular, pay attention to following areas of potential errors:

1. the preprocessing components of the source code. This is because the source code is generated from a template and may lack potential dataset specific preprocessing.
2. the scale of each input features

Once you have amended all the error message. Return the source code in  <source_code></source_code> tags

Tone Context: You are an intelligient machine learning expert, who knows about the dataset and the model used in the source code. The source code is facing some compilation/runtime error and you are trying to sort it out.

"""

