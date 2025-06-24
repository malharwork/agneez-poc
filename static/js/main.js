document.addEventListener('DOMContentLoaded', function() {
    const chapterTabBtn = document.getElementById('chapter-tab-btn');
    const pageTabBtn = document.getElementById('page-tab-btn');
    const chapterTab = document.getElementById('chapter-tab');
    const pageTab = document.getElementById('page-tab');
    const chapterItems = document.querySelectorAll('.chapter-item');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const contextText = document.getElementById('context-text');
    const loadingIndicator = document.getElementById('loading-indicator');
    const startPage = document.getElementById('start-page');
    const endPage = document.getElementById('end-page');
    const sectionTitle = document.getElementById('section-title');
    
    let currentContext = {
        type: null,
        chapter_number: null,
        start_page: null,
        end_page: null,
        title: null
    };
    
    // Tab switching
    chapterTabBtn.addEventListener('click', function() {
        chapterTab.classList.add('active');
        pageTab.classList.remove('active');
        chapterTabBtn.classList.add('active');
        pageTabBtn.classList.remove('active');
    });
    
    pageTabBtn.addEventListener('click', function() {
        pageTab.classList.add('active');
        chapterTab.classList.remove('active');
        pageTabBtn.classList.add('active');
        chapterTabBtn.classList.remove('active');
    });
    
    // Chapter selection
    chapterItems.forEach(item => {
        item.addEventListener('click', function() {
            // Remove selected class from all chapter items
            chapterItems.forEach(el => el.classList.remove('selected'));
            
            // Add selected class to clicked item
            this.classList.add('selected');
            
            // Get chapter info
            const chapterNumber = parseInt(this.dataset.chapter);
            const chapterTitle = this.querySelector('h5').textContent;
            
            // Update current context
            currentContext = {
                type: 'chapter',
                chapter_number: chapterNumber,
                title: chapterTitle,
                start_page: null,
                end_page: null
            };
            
            // Update context indicator
            contextText.textContent = `${chapterTitle}`;
            
            // Clear chat messages except the first one
            clearChatExceptFirst();
        });
    });
    
    // Page range selection
    const selectPageRange = document.createElement('button');
    selectPageRange.textContent = 'Select Pages';
    selectPageRange.classList.add('button');
    selectPageRange.style.marginTop = '1rem';
    selectPageRange.style.width = '100%';
    
    selectPageRange.addEventListener('click', function() {
        const start = parseInt(startPage.value);
        const end = parseInt(endPage.value);
        const title = sectionTitle.value || `Pages ${start}-${end}`;
        
        if (isNaN(start) || isNaN(end) || start < 1 || end < start) {
            alert('Please enter valid page numbers');
            return;
        }
        
        // Update current context
        currentContext = {
            type: 'page_range',
            start_page: start,
            end_page: end,
            title: title,
            chapter_number: null
        };
        
        // Update context indicator
        contextText.textContent = `${title} (Pages ${start}-${end})`;
        
        // Clear chat messages except the first one
        clearChatExceptFirst();
    });
    
    // Append select button to page tab
    pageTab.appendChild(selectPageRange);
    
    // Chat form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Check if context is selected
        if (!currentContext.type) {
            alert('Please select a chapter or page range first');
            return;
        }
        
        // Add user message to chat
        addMessage('user', message);
        
        // Clear input
        userInput.value = '';
        
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        
        // Prepare request data
        const requestData = {
            message: message,
            context_type: currentContext.type
        };
        
        if (currentContext.type === 'chapter') {
            requestData.chapter_number = currentContext.chapter_number;
        } else if (currentContext.type === 'page_range') {
            requestData.start_page = currentContext.start_page;
            requestData.end_page = currentContext.end_page;
        }
        
        // Send request to API
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            if (data.error) {
                addMessage('assistant', `Error: ${data.error}`);
            } else {
                addMessage('assistant', data.answer, data.source);
            }
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(error => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            // Add error message
            addMessage('assistant', `Error: ${error.message}`);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    });
    
    // Add message to chat
    function addMessage(role, content, source = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', role);
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.innerHTML = formatMessage(content);
        messageDiv.appendChild(contentDiv);
        
        if (source && role === 'assistant') {
            const sourceDiv = document.createElement('div');
            sourceDiv.classList.add('message-source');
            sourceDiv.textContent = `Source: ${source}`;
            messageDiv.appendChild(sourceDiv);
        }
        
        chatMessages.appendChild(messageDiv);
    }
    
    // Format message content with markdown-like syntax
    function formatMessage(content) {
        // Replace newlines with <br>
        content = content.replace(/\n/g, '<br>');
        
        // Bold text between ** **
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic text between * *
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        return content;
    }
    
    // Clear chat messages except the first one
    function clearChatExceptFirst() {
        const messages = chatMessages.querySelectorAll('.message');
        if (messages.length <= 1) return;
        
        for (let i = 1; i < messages.length; i++) {
            messages[i].remove();
        }
    }
    
    // Initial welcome message is already in the HTML
});