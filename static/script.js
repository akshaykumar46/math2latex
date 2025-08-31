class Whiteboard {
    constructor() {
        this.canvas = document.getElementById('whiteboard');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.currentTool = 'pencil';
        this.brushSize = 2;
        this.history = [];
        this.historyIndex = -1;
        this.maxHistory = 50;
        this.currentRequest = null; // For tracking current API request
        
        this.initializeCanvas();
        this.setupEventListeners();
        this.setupTools();
        this.saveState();
    }

    initializeCanvas() {
        // Set canvas background to white
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Set default drawing styles
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = this.brushSize;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));

        // Prevent context menu on right click
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    setupTools() {
        // Tool selection
        document.getElementById('pencil').addEventListener('click', () => this.selectTool('pencil'));
        document.getElementById('eraser').addEventListener('click', () => this.selectTool('eraser'));
        
        // Brush size control
        const brushSizeSlider = document.getElementById('brushSize');
        const brushSizeValue = document.getElementById('brushSizeValue');
        
        brushSizeSlider.addEventListener('input', (e) => {
            this.brushSize = parseInt(e.target.value);
            brushSizeValue.textContent = this.brushSize;
            this.updateBrushStyle();
        });

        // Action buttons
        document.getElementById('clear').addEventListener('click', () => this.clearCanvas());
        document.getElementById('undo').addEventListener('click', () => this.undo());
        
        // Convert button
        document.getElementById('convert').addEventListener('click', () => this.convertToLatex());
        
        // Stop button
        document.getElementById('stop').addEventListener('click', () => this.stopConversion());
        
        // Copy button
        document.getElementById('copyLatex').addEventListener('click', () => this.copyLatex());
    }

    selectTool(tool) {
        this.currentTool = tool;
        
        // Update active tool button
        document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById(tool).classList.add('active');
        
        this.updateBrushStyle();
    }

    updateBrushStyle() {
        if (this.currentTool === 'pencil') {
            this.ctx.strokeStyle = '#000000';
            this.ctx.globalCompositeOperation = 'source-over';
        } else if (this.currentTool === 'eraser') {
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.globalCompositeOperation = 'destination-out';
        }
        
        this.ctx.lineWidth = this.brushSize;
    }

    startDrawing(e) {
        this.isDrawing = true;
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.lastX = x;
        this.lastY = y;
    }

    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
        
        this.lastX = x;
        this.lastY = y;
    }

    stopDrawing() {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.saveState();
        }
    }

    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                        e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        
        this.canvas.dispatchEvent(mouseEvent);
    }

    saveState() {
        // Remove any states after current index
        this.history = this.history.slice(0, this.historyIndex + 1);
        
        // Add current state
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        this.history.push(imageData);
        this.historyIndex++;
        
        // Limit history size
        if (this.history.length > this.maxHistory) {
            this.history.shift();
            this.historyIndex--;
        }
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.ctx.putImageData(this.history[this.historyIndex], 0, 0);
        }
    }

    clearCanvas() {
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.saveState();
    }

    async convertToLatex() {
        const convertBtn = document.getElementById('convert');
        const stopBtn = document.getElementById('stop');
        const processingInfo = document.getElementById('processingInfo');
        const latexOutput = document.getElementById('latexOutput');
        
        // Show stop button and hide convert button
        convertBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        
        // Update UI state
        processingInfo.className = 'info-box loading';
        processingInfo.innerHTML = '<span class="spinner"></span>Processing your expression...';
        
        try {
            // Convert canvas to blob
            const blob = await new Promise(resolve => {
                this.canvas.toBlob(resolve, 'image/png');
            });
            
            // Create FormData for API request
            const formData = new FormData();
            formData.append('file', blob, 'expression.png');
            
            // Create AbortController for cancellation
            const controller = new AbortController();
            this.currentRequest = controller;
            
            // Make API request with abort signal
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            // Check if request was aborted
            if (controller.signal.aborted) {
                return;
            }
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Display result
            latexOutput.value = result.prediction;
            processingInfo.className = 'info-box success';
            processingInfo.innerHTML = `✅ Successfully converted! Processing time: ${result.processing_time.toFixed(3)}s`;
            
        } catch (error) {
            if (error.name === 'AbortError') {
                // Request was cancelled
                processingInfo.className = 'info-box error';
                processingInfo.innerHTML = '⏹️ Conversion cancelled by user';
                latexOutput.value = '';
            } else {
                console.error('Error converting expression:', error);
                processingInfo.className = 'info-box error';
                processingInfo.innerHTML = `❌ Error: ${error.message}`;
                latexOutput.value = '';
            }
        } finally {
            // Reset UI state
            this.currentRequest = null;
            convertBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        }
    }

    stopConversion() {
        if (this.currentRequest) {
            this.currentRequest.abort();
            this.currentRequest = null;
            
            const convertBtn = document.getElementById('convert');
            const stopBtn = document.getElementById('stop');
            const processingInfo = document.getElementById('processingInfo');
            
            // Reset UI state
            convertBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
            processingInfo.className = 'info-box error';
            processingInfo.innerHTML = '⏹️ Conversion cancelled by user';
        }
    }

    copyLatex() {
        const latexOutput = document.getElementById('latexOutput');
        if (latexOutput.value.trim()) {
            navigator.clipboard.writeText(latexOutput.value).then(() => {
                const copyBtn = document.getElementById('copyLatex');
                const originalText = copyBtn.textContent;
                copyBtn.textContent = '✅ Copied!';
                copyBtn.style.background = '#28a745';
                
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                    copyBtn.style.background = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                alert('Failed to copy text to clipboard');
            });
        }
    }

    // Method to get canvas as base64 string (alternative to blob)
    getCanvasAsBase64() {
        return this.canvas.toDataURL('image/png');
    }

    // Method to resize canvas (useful for different screen sizes)
    resizeCanvas(width, height) {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        tempCtx.drawImage(this.canvas, 0, 0);
        
        this.canvas.width = width;
        this.canvas.height = height;
        
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, width, height);
        
        this.ctx.drawImage(tempCanvas, 0, 0);
        this.saveState();
    }
}

// Initialize whiteboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.whiteboard = new Whiteboard();
    
    // Hide drawing hint after first interaction
    const canvas = document.getElementById('whiteboard');
    const hint = document.querySelector('.drawing-hint');
    
    const hideHint = () => {
        hint.style.opacity = '0';
        setTimeout(() => hint.style.display = 'none', 300);
    };
    
    canvas.addEventListener('mousedown', hideHint, { once: true });
    canvas.addEventListener('touchstart', hideHint, { once: true });
    
    // Responsive canvas sizing
    const resizeCanvas = () => {
        const container = canvas.parentElement;
        const maxWidth = Math.min(800, container.clientWidth - 40);
        const maxHeight = Math.min(400, window.innerHeight * 0.5);
        
        if (canvas.width !== maxWidth || canvas.height !== maxHeight) {
            window.whiteboard.resizeCanvas(maxWidth, maxHeight);
        }
    };
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
});

// Add some helpful keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
            case 'z':
                e.preventDefault();
                if (e.shiftKey) {
                    // Ctrl+Shift+Z or Cmd+Shift+Z for redo (not implemented yet)
                } else {
                    // Ctrl+Z or Cmd+Z for undo
                    window.whiteboard.undo();
                }
                break;
            case 'c':
                if (document.activeElement === document.getElementById('latexOutput')) {
                    // Let default copy behavior work for textarea
                    return;
                }
                e.preventDefault();
                window.whiteboard.copyLatex();
                break;
        }
    }
    
    // Spacebar to convert
    if (e.code === 'Space' && document.activeElement !== document.getElementById('latexOutput')) {
        e.preventDefault();
        window.whiteboard.convertToLatex();
    }
    
    // Escape key to stop conversion
    if (e.code === 'Escape' && window.whiteboard.currentRequest) {
        e.preventDefault();
        window.whiteboard.stopConversion();
    }
});
