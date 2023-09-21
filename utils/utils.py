def hidden_to_heads(x, config):
    num_attention_heads = config.attention_config.num_heads
    attention_head_size =  config.attention_config.d_model // config.attention_config.num_heads
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(new_x_shape)
    return x