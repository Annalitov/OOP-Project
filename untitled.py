import telebot
API_TOKEN = '5867678740:AAG1LP0N37Y9v6vYuXd3nfi-rOTrcLSccOM'
bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(command = ['start'])
def start(message):
	#mess = f'Hello, {message.from_user.first_name}'
	bot.send_message(message.chat.id, 'mess') 
bot.polling(none_stop=True)



import telebot

bot = telebot.TeleBot('5867678740:AAG1LP0N37Y9v6vYuXd3nfi-rOTrcLSccOM')

@bot.message_handler(command = ['start'])
def start(message):
	bot.send_message(message.chat.id, 'Hello') 
bot.polling(none_stop=True)