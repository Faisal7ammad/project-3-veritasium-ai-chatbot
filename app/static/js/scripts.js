document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('send-button').addEventListener('click', sendMessage);

    document.getElementById('chat-input').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        const inputBox = document.getElementById('chat-input');
        const message = inputBox.value.trim();

        if (message === '') return;

        console.log("User message: ", message); // Debugging

        const chatBox = document.getElementById('chat-box');
        const userMessage = document.createElement('div');
        userMessage.className = 'message user-message';
        userMessage.textContent = 'You: ' + message;
        chatBox.appendChild(userMessage);
        inputBox.value = '';

        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: message, context: '' })
        })
        .then(response => {
            console.log("Response status: ", response.status); // Debugging
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log("Response data: ", data); // Debugging
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot-message';
            botMessage.innerHTML = 'Bot: ' + data.response.replace(/\n/g, '<br>');
            chatBox.appendChild(botMessage);
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot-message';
            botMessage.textContent = 'Bot: An error occurred. Please try again.';
            chatBox.appendChild(botMessage);
            chatBox.scrollTop = chatBox.scrollHeight;
        });
    }
});
