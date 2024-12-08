css = '''
<style>
.chat-message {
  padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem ;    display:flex;
}
.chat-message.user {
  background-color: #aec6cf;
}
.chat-message.bot {
  background-color: #e9eaec
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 30px;
  max-height: 30px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #000000;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
    <a href="https://imgbb.com/"><img src="https://i.ibb.co/1XLGTs1/Whats-App-Image-2024-11-21-at-20-26-32-5dec28a6.jpg" alt="Whats-App-Image-2024-11-21-at-20-26-32-5dec28a6" border="0" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;"></a>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/RcCJZ48/Whats-App-Image-2024-11-23-at-00-28-22-8496dfbe.jpg" alt="Whats-App-Image-2024-11-23-at-00-28-22-8496dfbe" border="0" style="max-height: 30px; max-width: 30px; border-radius: 50%; object-fit: cover; ">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''