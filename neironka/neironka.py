from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pika
from functools import partial
import logging
import os
import json

#torch.cuda.is_available()
#!nvidia-smi

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_name_or_path = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)


from transformers import TextDataset, DataCollatorForLanguageModeling

# Сохраним обучающие данные в .txt файл 
train_path = 'jokes_B.txt'


# Создание датасета
train_dataset = TextDataset(tokenizer=tokenizer,file_path=train_path,block_size=32)
  
# Создание даталодера (нарезает текст на оптимальные по длине куски)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./Desktop1", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=200, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=32,  # batch size for evaluation
    warmup_steps=10,# number of warmup steps for learning rate scheduler
    gradient_accumulation_steps=16, # to make "virtual" batch size larger
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers = (torch.optim.AdamW(model.parameters(),lr=1e-5),None) # Optimizer and lr scheduler
)

# print(len(train_dataset))

#trainer.train()

# Пример вероятностного сэмплирвоания с ограничением
global answer
#text = 'Доктор'
def generation(model, tokenizer, text): #нужно дописать аргументы функции
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids, 
                            do_sample=True,
                            num_beams=2,
                            temperature=1.5,
                            top_p=0.9,
                            max_length=50,
                            )
    
    answer = list(map(tokenizer.decode, out))[0]
    return str(answer)
#print()
#print(answer)
 
def json_deserializer(m):
    return json.loads(m.decode('utf-8'))


def json_serializer(data):
    return json.dumps(data).encode("utf-8")

amqp_url = os.environ["AMQP_URL"]

connection_parameters = pika.URLParameters(amqp_url)
connection = pika.BlockingConnection(connection_parameters)

channel = connection.channel()
channel2 = connection.channel()

channel.queue_declare(queue='request-queue')
channel2.queue_declare(queue='reply-queue')

def on_request_message_received(ch, method, properties, body):
    body_str = json_deserializer(body)
    logging.info(f"Received Request: {body_str}")
    answer = generation(model, tokenizer, body_str['text'])
    reply = {'chat_id': body_str['chat_id'], 'answer': answer}
    reply = json_serializer(reply)
    logging.info("answer ", reply)
    channel2.basic_publish('', routing_key='reply-queue', body=reply)

def main():
    channel.basic_consume(queue='request-queue', auto_ack=True, on_message_callback=on_request_message_received)
    channel.start_consuming()

if __name__ == '__main__':
    main()