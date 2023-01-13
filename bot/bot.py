import telebot, pika
from telebot import types
keyboard1 = telebot.types.ReplyKeyboardMarkup()
keyboard1.row('Получить шутку', 'Закончить')

API_TOKEN = '5867678740:AAG1LP0N37Y9v6vYuXd3nfi-rOTrcLSccOM'
bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(commands=['start'])
def start_message(message):
    #name = message.from_user.first_name
    #user_full_name = message.from_user.full_name
    bot.send_message(message.chat.id, 'Привет, хочешь начать /start', reply_markup=keyboard1)
    
@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text.lower() =='Hel':
        bot.send_message(message.chat.id, message.text.upper() )
    elif message.text =='Закончить':
        bot.send_message(message.chat.id,'Пока, до новых втреч' )
    elif message.text =='Получить шутку':
        bot.send_message(message.chat.id,'Введите начало шутки' )
        global u_text
        u_text = message.text
        




answer = ''
@bot.message_handler(commands=['шутка'])
async def start_handler(message: types.Message):
    letters = 'йцукенгшщзхъфывапролджэячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ'
    prompt = '<s>' + ''.join(random.choice(letters) for i in range(random.randint(2, 6))) + 'Шутка: '
    
    ch.basic_publish('', routing_key=properties.reply_to, body=u_text)
    connection_parameters = pika.ConnectionParameters('rabbit')
    connection = pika.BlockingConnection(connection_parameters)
    #функция отправки прописать чтобы отпралялся запрос
    def on_reply_message_received(ch, method, properties, body):
        ch.stop_consuming()
        body_str = body.decode("utf-8")
        print(f"Received: {body_str}")
        global answer
        answer = body_str
        print(f"Answer: {answer}")

    channel = connection.channel()
    reply_queue = channel.queue_declare(queue='', exclusive=True)
    channel.basic_consume(queue=reply_queue.method.queue,
                          on_message_callback=on_reply_message_received)
    channel.queue_declare(queue='request-queue')

    cor_id = str(uuid.uuid4())
    print(f"Sending Request: {prompt}")

    channel.basic_publish('', routing_key='request-queue', properties=pika.BasicProperties(
        reply_to=reply_queue.method.queue,
        correlation_id=cor_id
    ), body=prompt)

    print("Starting Client")

    channel.start_consuming()
    global answer
    await message.answer(answer)

if __name__ == '__main__':
    bot.polling(none_stop=True)
    