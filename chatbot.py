import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_chat_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def get_response(user_input: str, chat_history_ids = None) -> tuple:
    """
    Generates a chatbot reply given user input and previous conversation history.

    Args:
        user_input      : latest message typed by the user
        chat_history_ids: tensor of previous conversation tokens (or None for first message)

    Returns:
        (reply_string, updated_history_tensor)
    """
    tokenizer, model = load_chat_model()

    new_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token,
            return_tensors ="pt"
        )

    if chat_history_ids is not None and len(chat_history_ids) > 0:
         history_tensor = torch.cat([chat_history_ids, new_input_ids], dim =-1)

    else:
        history_tensor =new_input_ids

    if history_tensor.shape[-1] >512:
        history_tensor = history_tensor[:,-512:]

    with torch.no_grad():
        bot_output = model.generate(
            history_tensor,
            max_length = history_tensor.shape[-1]+ 100,
            pad_token_id = tokenizer.eos_token_id,
            do_sample = True,
            top_k = 50,
            top_p = 0.95,
            temperature = 0.75,
            repetetion_penalty = 1.3,
        ) 
    
    response = tokenizer.decode(
        bot_output[:, history_tensor.shape[-1]:][0],
        skip_special_tokens = True
    )

    if not response_strip():
        response = "I'm not sure how to respond to that. Could you rephrase?"
    return response, bot_output
    