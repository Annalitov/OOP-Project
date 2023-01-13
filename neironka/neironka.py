
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pika
from functools import partial
#torch.cuda.is_available()
#!nvidia-smi
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)


from transformers import TextDataset, DataCollatorForLanguageModeling

# Сохраним обучающие данные в .txt файл 
train_path = 'C:/Users/Anna/Desktop/oop/jokes_B.txt'


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
text = ''


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

generated_text = list(map(tokenizer.decode, out))[0]
#print()
#print(generated_text)


def main():

    def on_request_message_received(model, ch, method, properties, body):
        body_str = body.decode("utf-8")
        print(f"Received Request: {body_str}")
        answer = generated_text
        print("answer ", answer)
        ch.basic_publish('', routing_key=properties.reply_to, body=answer)

    connection_parameters = pika.ConnectionParameters('rabbit')
    connection = pika.BlockingConnection(connection_parameters)

    channel = connection.channel()
    channel.queue_declare(queue='request-queue')
    channel.basic_consume(queue='request-queue', auto_ack=True,
                          on_message_callback=partial(on_request_message_received, model))

    channel.basic_publish('', routing_key='request-queue', properties=pika.BasicProperties(
        reply_to=reply_queue.method.queue,
        correlation_id=cor_id
    ), body=prompt)
    
    print("Starting Server")

    channel.start_consuming()


if __name__ == '__main__':
    main()
       