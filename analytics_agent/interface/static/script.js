const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const resetButton = document.getElementById('resetButton');

let currentEventSource = null;
let currentAssistantMessage = null;
let isProcessing = false;

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// Send message on Enter (Shift+Enter for new line)
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!isProcessing) {
            sendMessage();
        }
    }
});

sendButton.addEventListener('click', sendMessage);
resetButton.addEventListener('click', resetConversation);

function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    // Add user message to chat
    addMessage('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Disable input
    isProcessing = true;
    sendButton.disabled = true;
    messageInput.disabled = true;

    // Create new assistant message container
    currentAssistantMessage = createAssistantMessageContainer();

    // Show typing indicator inside the assistant message
    showTypingIndicator();

    // Start streaming response using fetch
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'same-origin', // Include cookies for session
        body: JSON.stringify({ message: message }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    hideTypingIndicator();
                    isProcessing = false;
                    sendButton.disabled = false;
                    messageInput.disabled = false;
                    messageInput.focus();
                    currentEventSource = null;
                    currentAssistantMessage = null;
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            switch (data.type) {
                                case 'agent_thinking':
                                    addToMessage(currentAssistantMessage, 'thinking', data.content);
                                    break;
                                case 'tool_execution_start':
                                    // Tool execution started
                                    break;
                                case 'tool_call':
                                    addToMessage(currentAssistantMessage, 'tool-call', data.content);
                                    break;
                                case 'tool_result':
                                    addToMessage(currentAssistantMessage, 'tool-result', data.content);
                                    break;
                                case 'visualization':
                                    addVisualization(currentAssistantMessage, data.url, data.filename);
                                    break;
                                case 'response':
                                    addToMessage(currentAssistantMessage, 'response', data.content);
                                    break;
                                case 'error':
                                    addToMessage(currentAssistantMessage, 'error', `Error: ${data.content}`);
                                    break;
                                case 'done':
                                    hideTypingIndicator();
                                    isProcessing = false;
                                    sendButton.disabled = false;
                                    messageInput.disabled = false;
                                    messageInput.focus();
                                    currentEventSource = null;
                                    currentAssistantMessage = null;
                                    return;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }

                readStream();
            }).catch(error => {
                console.error('Stream reading error:', error);
                hideTypingIndicator();
                addToMessage(currentAssistantMessage, 'error', 'Connection error. Please try again.');
                isProcessing = false;
                sendButton.disabled = false;
                messageInput.disabled = false;
                currentEventSource = null;
            });
        }

        readStream();
    })
    .catch(error => {
        console.error('Fetch error:', error);
        hideTypingIndicator();
        addToMessage(currentAssistantMessage, 'error', 'Connection error. Please try again.');
        isProcessing = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        currentEventSource = null;
    });
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function createAssistantMessageContainer() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.id = 'current-assistant-message';

    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();

    return contentDiv;
}

function addToMessage(container, type, content) {
    if (!container) return;

    let element;
    if (type === 'response') {
        // Format response with markdown-like formatting
        element = document.createElement('div');
        element.innerHTML = formatMessage(content);
    } else {
        element = document.createElement('div');
        element.className = type;
        element.textContent = content;
    }

    // Insert before typing indicator if it exists, otherwise append
    const typingIndicator = container.querySelector('.typing-indicator');
    if (typingIndicator) {
        container.insertBefore(element, typingIndicator);
    } else {
        container.appendChild(element);
    }
    scrollToBottom();
}

function addVisualization(container, url, filename) {
    if (!container) return;

    const visualizationDiv = document.createElement('div');
    visualizationDiv.className = 'visualization';

    const img = document.createElement('img');
    img.src = url;
    img.alt = filename;
    img.className = 'visualization-image';
    img.loading = 'lazy';

    visualizationDiv.appendChild(img);

    // Insert before typing indicator if it exists, otherwise append
    const typingIndicator = container.querySelector('.typing-indicator');
    if (typingIndicator) {
        container.insertBefore(visualizationDiv, typingIndicator);
    } else {
        container.appendChild(visualizationDiv);
    }
    scrollToBottom();
}

function formatMessage(text) {
    // Use marked.js to parse markdown
    try {
        // Configure marked options
        marked.setOptions({
            breaks: true,  // Convert \n to <br>
            gfm: true,     // GitHub Flavored Markdown
            headerIds: false,  // Disable header IDs
            mangle: false,  // Don't escape email addresses
            highlight: function(code, lang) {
                // Syntax highlighting using highlight.js
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.error('Highlight error:', err);
                    }
                }
                // Auto-detect language if not specified
                try {
                    return hljs.highlightAuto(code).value;
                } catch (err) {
                    console.error('Highlight auto error:', err);
                }
                return code;
            }
        });

        // Parse markdown to HTML
        const html = marked.parse(text);
        return html;
    } catch (error) {
        console.error('Error parsing markdown:', error);
        // Fallback to simple formatting
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    }
}

function showTypingIndicator() {
    if (!currentAssistantMessage) return;

    // Remove existing typing indicator if present
    const existing = currentAssistantMessage.querySelector('.typing-indicator');
    if (existing) {
        existing.remove();
    }

    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;

    currentAssistantMessage.appendChild(indicator);
    scrollToBottom();
}

function hideTypingIndicator() {
    if (!currentAssistantMessage) return;

    const indicator = currentAssistantMessage.querySelector('.typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function resetConversation() {
    if (isProcessing) {
        // Cancel any ongoing requests
        hideTypingIndicator();
        isProcessing = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        currentEventSource = null;
    }

    try {
        const response = await fetch('/api/reset', {
            method: 'POST',
            credentials: 'same-origin', // Include cookies for session
        });

        if (response.ok) {
            // Clear chat container except welcome message
            const welcomeMessage = chatContainer.querySelector('.welcome-message');
            chatContainer.innerHTML = '';
            if (welcomeMessage) {
                chatContainer.appendChild(welcomeMessage);
            } else {
                // Recreate welcome message if it was removed
                const welcomeDiv = document.createElement('div');
                welcomeDiv.className = 'welcome-message';
                welcomeDiv.innerHTML = `
                    <p class="welcome-title">Welcome to Analytics Agent</p>
                    <p class="welcome-text">I can help you query and analyze your BigQuery data, create visualizations, and export data to files.</p>
                    <p class="welcome-text">Start by asking a question about your data.</p>
                `;
                chatContainer.appendChild(welcomeDiv);
            }
            scrollToBottom();
        }
    } catch (error) {
        console.error('Error resetting conversation:', error);
    }
}

// Focus input on load
messageInput.focus();

