document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.querySelector('.chat-container');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const newChatBtn = document.querySelector('.new-chat-btn');
    const clearHistoryBtn = document.querySelector('.clear-history-btn');
    const deleteChatBtn = document.querySelector('.delete-chat-btn');
    const chatHistory = document.querySelector('.chat-history');
    let currentChatId = null;
    let charts = {};

    // Load chat history on startup
    loadChatHistory();

    // New chat button handler
    newChatBtn.addEventListener('click', () => {
        currentChatId = null;
        clearChat();
        updateChatTitle('New Chat');
        setActiveChat(null);
    });

    // Clear history button handler
    clearHistoryBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear all chat history? This action cannot be undone.')) {
            try {
                const response = await fetch('/chat_history/clear', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    // Clear current chat
                    currentChatId = null;
                    clearChat();
                    updateChatTitle('New Chat');
                    
                    // Reload chat history (should be empty)
                    loadChatHistory();
                    
                    // Show success message
                    appendMessage('assistant', 'Chat history has been cleared successfully.');
                } else {
                    throw new Error(data.message);
                }
            } catch (error) {
                console.error('Error clearing chat history:', error);
                appendMessage('assistant', 'Error clearing chat history. Please try again.');
            }
        }
    });

    // Delete current chat button handler
    deleteChatBtn.addEventListener('click', async () => {
        if (!currentChatId) {
            appendMessage('assistant', 'No chat selected to delete.');
            return;
        }

        if (confirm('Are you sure you want to delete this chat?')) {
            try {
                const response = await fetch(`/chat_history/${currentChatId}`, {
                    method: 'DELETE'
                });
                
                // Clear current chat
                currentChatId = null;
                clearChat();
                updateChatTitle('New Chat');
                
                // Reload chat history
                loadChatHistory();
                
                // Show success message
                appendMessage('assistant', 'Chat has been deleted successfully.');
            } catch (error) {
                console.error('Error deleting chat:', error);
                appendMessage('assistant', 'Error deleting chat. Please try again.');
            }
        }
    });

    // Delete individual message handler
    function deleteMessage(timestamp) {
        if (confirm('Are you sure you want to delete this message?')) {
            fetch(`/chat_history/entry/${timestamp}`, {
                method: 'DELETE'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Reload the current chat to reflect changes
                    if (currentChatId) {
                        loadChat(currentChatId);
                    }
                } else {
                    throw new Error(data.message);
                }
            })
            .catch(error => {
                console.error('Error deleting message:', error);
                appendMessage('assistant', 'Error deleting message. Please try again.');
            });
        }
    }

    messageForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to UI
        appendMessage('user', message);
        messageInput.value = '';

        // Show typing indicator
        const typingIndicator = appendTypingIndicator();

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query: message,
                    chat_id: currentChatId
                })
            });

            const data = await response.json();
            
            // Remove typing indicator
            typingIndicator.remove();

            if (data.error) {
                appendMessage('assistant', `Error: ${data.error}`);
                return;
            }

            // If this is a new chat, create chat history entry and update UI
            if (!currentChatId) {
                currentChatId = data.message.chat_id;
                updateChatTitle(data.message.title || 'New Chat');
                loadChatHistory();
            }

            // Handle the formatted response
            appendFormattedResponse(data.formatted_response);

        } catch (error) {
            console.error('Error:', error);
            typingIndicator.remove();
            appendMessage('assistant', 'Sorry, there was an error processing your request.');
        }
    });

    async function loadChatHistory() {
        try {
            const response = await fetch('/chat_history');
            const data = await response.json();
            
            chatHistory.innerHTML = '';
            data.chats.forEach(chat => {
                const chatItem = createChatHistoryItem(chat);
                chatHistory.appendChild(chatItem);
            });
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    async function createChatHistoryEntry(chatId, firstMessage) {
        try {
            const response = await fetch('/messages/' + chatId);
            const data = await response.json();
            const messages = data.messages || [];
            
            // Get the last few messages to create a meaningful title
            const lastMessages = messages.slice(-3);
            let title = '';
            
            if (lastMessages.length > 0) {
                // Use the first user message as the title
                const userMessage = lastMessages.find(m => m.role === 'user');
                if (userMessage) {
                    title = userMessage.content;
                } else {
                    title = lastMessages[0].content;
                }
            } else {
                title = firstMessage;
            }
            
            // Truncate title if too long
            title = title.substring(0, 50) + (title.length > 50 ? '...' : '');
            
            await fetch('/chat_history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    chat_id: chatId,
                    title: title
                })
            });
        } catch (error) {
            console.error('Error creating chat history entry:', error);
        }
    }

    async function saveMessage(role, content) {
        try {
            const response = await fetch('/messages', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    chat_id: currentChatId,
                    role: role,
                    content: content
                })
            });
            return await response.json();
        } catch (error) {
            console.error('Error saving message:', error);
            return null;
        }
    }

    async function loadChat(chatId) {
        try {
            const response = await fetch(`/messages/${chatId}`);
            const data = await response.json();
            
            clearChat();
            data.messages.forEach(message => {
                if (message.role === 'assistant' && message.formatted_response) {
                    appendFormattedResponse(message.formatted_response);
                } else {
                    appendMessage(message.role, message.content);
                }
            });
            
            currentChatId = chatId;
        } catch (error) {
            console.error('Error loading chat:', error);
        }
    }

    function createChatHistoryItem(chat) {
        const item = document.createElement('div');
        item.className = 'chat-history-item';
        
        const title = document.createElement('div');
        title.className = 'title';
        title.textContent = chat.title;
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
        
        // Add click event for delete button
        deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation(); // Prevent chat from being loaded when deleting
            
            if (confirm('Are you sure you want to delete this chat?')) {
                try {
                    await fetch(`/chat_history/${chat.id}`, {
                        method: 'DELETE'
                    });
                    
                    // If the deleted chat was the current chat, create a new chat
                    if (currentChatId === chat.id) {
                        currentChatId = null;
                        clearChat();
                        updateChatTitle('New Chat');
                    }
                    
                    // Reload chat history
                    loadChatHistory();
                } catch (error) {
                    console.error('Error deleting chat:', error);
                }
            }
        });
        
        item.appendChild(title);
        item.appendChild(deleteBtn);
        
        // Add click event for loading chat
        item.addEventListener('click', (e) => {
            if (e.target !== deleteBtn && !deleteBtn.contains(e.target)) {
                loadChat(chat.id);
                updateChatTitle(chat.title);
                setActiveChat(item);
            }
        });
        
        return item;
    }

    function clearChat() {
        chatContainer.innerHTML = '';
        appendMessage('assistant', 'Hello! How can I assist you today? If you have any questions about medical supplies, inventory, or transportation protocols, feel free to ask!');
    }

    function updateChatTitle(title) {
        document.querySelector('.chat-title').textContent = title;
    }

    function setActiveChat(item) {
        document.querySelectorAll('.chat-history-item').forEach(i => i.classList.remove('active'));
        if (item) item.classList.add('active');
    }

    function appendFormattedResponse(formattedResponse) {
        const responseContainer = document.createElement('div');
        responseContainer.className = 'assistant-message chat-message';

        // Create markdown content container
        const markdownContainer = document.createElement('div');
        markdownContainer.className = 'markdown-content';

        // Combine all parts of the response into a markdown string
        let markdownContent = '';

        // Add summary section
        if (formattedResponse.summary) {
            markdownContent += formattedResponse.summary + '\n\n';
        }

        // Add structured data if present
        if (formattedResponse.structured_data) {
            markdownContent += '### Structured Data\n';
            if (Array.isArray(formattedResponse.structured_data)) {
                formattedResponse.structured_data.forEach(item => {
                    markdownContent += `- ${item}\n`;
                });
            } else {
                markdownContent += JSON.stringify(formattedResponse.structured_data, null, 2);
            }
            markdownContent += '\n\n';
        }

        // Add insights if present
        if (formattedResponse.insights && formattedResponse.insights.length > 0) {
            markdownContent += '### Key Insights\n';
            formattedResponse.insights.forEach(insight => {
                markdownContent += `#### ${insight.title}\n`;
                markdownContent += `${insight.description}\n\n`;
                if (insight.recommendation) {
                    markdownContent += `**Recommendation:** ${insight.recommendation}\n`;
                }
                if (insight.priority) {
                    markdownContent += `**Priority:** ${insight.priority}\n`;
                }
                if (insight.metrics) {
                    markdownContent += '\n**Metrics:**\n';
                    Object.entries(insight.metrics).forEach(([key, value]) => {
                        markdownContent += `- ${key}: ${value}\n`;
                    });
                }
                markdownContent += '\n';
            });
        }

        // Add trends if present
        if (formattedResponse.trends) {
            markdownContent += '### Trends\n';
            if (Array.isArray(formattedResponse.trends)) {
                formattedResponse.trends.forEach(trend => {
                    markdownContent += `- ${trend}\n`;
                });
            } else {
                markdownContent += formattedResponse.trends + '\n';
            }
            markdownContent += '\n';
        }

        // Parse and render markdown
        markdownContainer.innerHTML = marked.parse(markdownContent);
        responseContainer.appendChild(markdownContainer);

        // Add tables if present
        if (formattedResponse.tables && formattedResponse.tables.length > 0) {
            const tablesContainer = document.createElement('div');
            tablesContainer.className = 'tables-container mt-3';
            formattedResponse.tables.forEach(tableData => {
                const tableWrapper = document.createElement('div');
                tableWrapper.className = 'table-responsive mt-3';
                const table = createDataTable(tableData);
                tableWrapper.appendChild(table);
                tablesContainer.appendChild(tableWrapper);
            });
            responseContainer.appendChild(tablesContainer);
        }

        // Add charts if present
        if (formattedResponse.charts && formattedResponse.charts.length > 0) {
            const chartsContainer = document.createElement('div');
            chartsContainer.className = 'charts-grid mt-3';
            
            formattedResponse.charts.forEach((chartData, index) => {
                const chartWrapper = document.createElement('div');
                chartWrapper.className = 'chart-wrapper';
                
                if (chartData.description) {
                    const description = document.createElement('div');
                    description.className = 'chart-description';
                    description.textContent = chartData.description;
                    chartWrapper.appendChild(description);
                }

                const chartContainer = document.createElement('div');
                chartContainer.className = 'chart-container';
                const canvas = document.createElement('canvas');
                canvas.id = `chart-${Date.now()}-${index}`;
                chartContainer.appendChild(canvas);
                chartWrapper.appendChild(chartContainer);
                chartsContainer.appendChild(chartWrapper);

                // Create chart
                const ctx = canvas.getContext('2d');
                const chart = new Chart(ctx, chartData);
            });
            
            responseContainer.appendChild(chartsContainer);
        }

        chatContainer.appendChild(responseContainer);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function createInsightCard(insight) {
        const card = document.createElement('div');
        card.className = `insight-card priority-${insight.priority}`;

        const title = document.createElement('h4');
        title.className = 'insight-title';
        title.textContent = insight.title;

        const description = document.createElement('p');
        description.className = 'insight-description';
        description.textContent = insight.description;

        const recommendation = document.createElement('div');
        recommendation.className = 'insight-recommendation';
        recommendation.innerHTML = `<strong>Recommendation:</strong> ${insight.recommendation}`;

        const metrics = document.createElement('div');
        metrics.className = 'insight-metrics';
        Object.entries(insight.metrics).forEach(([key, value]) => {
            const metric = document.createElement('div');
            metric.className = 'metric';
            metric.innerHTML = `<span class="metric-label">${key}:</span> <span class="metric-value">${value}</span>`;
            metrics.appendChild(metric);
        });

        card.appendChild(title);
        card.appendChild(description);
        card.appendChild(recommendation);
        card.appendChild(metrics);

        return card;
    }

    function createTrendsSection(trends) {
        const container = document.createElement('div');
        container.className = 'trends-container mt-3';

        const title = document.createElement('h4');
        title.className = 'trends-title';
        title.textContent = 'Trend Analysis';

        const metrics = document.createElement('div');
        metrics.className = 'trends-metrics';

        Object.entries(trends).forEach(([key, value]) => {
            if (key !== 'timestamp') {
                const metric = document.createElement('div');
                metric.className = 'trend-metric';
                const formattedValue = typeof value === 'number' ? 
                    value.toFixed(2) : value;
                metric.innerHTML = `
                    <span class="metric-label">${key.replace(/_/g, ' ').toUpperCase()}:</span>
                    <span class="metric-value">${formattedValue}</span>
                `;
                metrics.appendChild(metric);
            }
        });

        container.appendChild(title);
        container.appendChild(metrics);

        return container;
    }

    function createDataTable(tableData) {
        const table = document.createElement('table');
        table.className = 'table table-dark table-striped table-hover';
        
        // Create header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        tableData.headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create body
        const tbody = document.createElement('tbody');
        tableData.rows.forEach(row => {
            const tr = document.createElement('tr');
            tableData.headers.forEach(header => {
                const td = document.createElement('td');
                const key = header.toLowerCase().replace(/ /g, '_');
                td.textContent = row[key];
                
                // Add status styling
                if (header === 'Status') {
                    td.className = `status-${row[key]}`;
                }
                // Add risk level styling
                if (header === 'Risk Level') {
                    td.className = `risk-${row[key].toLowerCase().replace(' ', '-')}`;
                }
                
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        return table;
    }

    function appendTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator assistant-message chat-message';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        chatContainer.appendChild(indicator);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return indicator;
    }

    function appendMessage(role, content, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${role}-message chat-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        
        if (timestamp) {
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-msg-btn';
            deleteBtn.innerHTML = '<i class="fas fa-times"></i>';
            deleteBtn.onclick = () => deleteMessage(timestamp);
            actionsDiv.appendChild(deleteBtn);
        }
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(actionsDiv);
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageDiv;
    }
}); 