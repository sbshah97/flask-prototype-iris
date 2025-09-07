class NITKChatApp {
    constructor() {
        this.chatContainer = document.getElementById('chatContainer');
        this.questionInput = document.getElementById('questionInput');
        this.sendButton = document.getElementById('sendButton');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.inputHelp = document.getElementById('inputHelp');
        
        this.isLoading = false;
        this.apiBase = window.location.origin;
        
        this.init();
    }
    
    init() {
        // Event listeners
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.questionInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.questionInput.addEventListener('input', () => this.handleInputChange());
        
        // Auto-resize textarea
        this.questionInput.addEventListener('input', () => this.autoResizeTextarea());
        
        // Check initial status
        this.checkStatus();
        
        // Focus on input
        this.questionInput.focus();
    }
    
    handleKeyDown(e) {
        if (e.key === 'Enter') {
            if (e.shiftKey) {
                // Allow new line with Shift+Enter
                return;
            } else {
                // Send message with Enter
                e.preventDefault();
                this.sendMessage();
            }
        }
    }
    
    handleInputChange() {
        const question = this.questionInput.value.trim();
        this.sendButton.disabled = !question || this.isLoading;
        
        // Update help text based on content
        if (question) {
            this.inputHelp.textContent = 'Press Enter to send, Shift+Enter for new line';
        } else {
            this.inputHelp.textContent = 'Type your question about NITK policies...';
        }
    }
    
    autoResizeTextarea() {
        const textarea = this.questionInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    async checkStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            if (data.pipeline_initialized) {
                this.updateStatus('ready', 'Ready to help!');
            } else {
                this.updateStatus('loading', 'Initializing...');
            }
        } catch (error) {
            console.error('Status check failed:', error);
            this.updateStatus('error', 'Connection error');
        }
    }
    
    updateStatus(type, message) {
        this.statusIndicator.className = `status-indicator ${type}`;
        this.statusText.textContent = message;
    }
    
    async sendMessage() {
        const question = this.questionInput.value.trim();
        if (!question || this.isLoading) return;
        
        // Add user message to chat
        this.addMessage(question, 'user');
        
        // Clear input and disable button
        this.questionInput.value = '';
        this.autoResizeTextarea();
        this.setLoading(true);
        
        // Add loading message
        const loadingMessageElement = this.addLoadingMessage();
        
        try {
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });
            
            const data = await response.json();
            
            // Remove loading message
            loadingMessageElement.remove();
            
            if (response.ok) {
                // Add assistant response
                this.addMessage(data.answer, 'assistant', data.sources);
                this.updateStatus('ready', 'Ready to help!');
            } else {
                // Add error message
                this.addErrorMessage(data.error || 'An error occurred');
                this.updateStatus('error', 'Error occurred');
            }
        } catch (error) {
            console.error('Chat request failed:', error);
            loadingMessageElement.remove();
            this.addErrorMessage('Failed to connect to the server. Please try again.');
            this.updateStatus('error', 'Connection failed');
        }
        
        this.setLoading(false);
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        this.sendButton.disabled = loading || !this.questionInput.value.trim();
        this.questionInput.disabled = loading;
        
        if (loading) {
            this.inputHelp.textContent = 'Processing your question...';
        } else {
            this.handleInputChange();
            this.questionInput.focus();
        }
    }
    
    addMessage(content, type, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (type === 'assistant') {
            // Format assistant message with proper line breaks
            messageContent.innerHTML = this.formatAssistantMessage(content);
            
            // Add sources if available
            if (sources && sources.length > 0) {
                const sourcesDiv = this.createSourcesDiv(sources);
                messageContent.appendChild(sourcesDiv);
            }
        } else {
            messageContent.textContent = content;
        }
        
        messageDiv.appendChild(messageContent);
        
        // Remove welcome message if it exists
        const welcomeMessage = this.chatContainer.querySelector('.welcome-message');
        if (welcomeMessage && type === 'user') {
            welcomeMessage.remove();
        }
        
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    addLoadingMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content loading';
        messageContent.innerHTML = `
            Thinking
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        messageDiv.appendChild(messageContent);
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    addErrorMessage(error) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content error-message';
        messageContent.textContent = `Error: ${error}`;
        
        messageDiv.appendChild(messageContent);
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    formatAssistantMessage(content) {
        // Convert line breaks to HTML and handle basic formatting
        return content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>')
            .replace(/<p><\/p>/g, '');
    }
    
    createSourcesDiv(sources) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        
        const title = document.createElement('h4');
        title.textContent = `📚 Sources (${sources.length})`;
        sourcesDiv.appendChild(title);
        
        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const header = document.createElement('div');
            header.className = 'source-header';
            header.textContent = `${source.handbook} Handbook - ${source.source_file} (Page ${source.page})`;
            
            const snippet = document.createElement('div');
            snippet.className = 'source-snippet';
            snippet.textContent = source.snippet + (source.snippet.length >= 300 ? '...' : '');
            
            sourceItem.appendChild(header);
            sourceItem.appendChild(snippet);
            sourcesDiv.appendChild(sourceItem);
        });
        
        return sourcesDiv;
    }
    
    scrollToBottom() {
        requestAnimationFrame(() => {
            this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        });
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new NITKChatApp();
});

// Sample questions for demo
const sampleQuestions = [
    "What are the attendance requirements for BTech students?",
    "How is the CGPA calculated?",
    "What is the minimum credit requirement for graduation?",
    "What are the eligibility criteria for semester promotion?",
    "How many backlogs are allowed in each semester?"
];

// Add sample questions to the page (optional)
function addSampleQuestions() {
    const container = document.querySelector('.welcome-message .message-content');
    if (container) {
        const samplesDiv = document.createElement('div');
        samplesDiv.innerHTML = `
            <p><strong>Sample questions you can ask:</strong></p>
            <ul>
                ${sampleQuestions.map(q => `<li>"${q}"</li>`).join('')}
            </ul>
        `;
        container.appendChild(samplesDiv);
    }
}