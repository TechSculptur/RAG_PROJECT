const API_URL = "http://127.0.0.1:8000";

// Stores conversation history for the current browser session
let chatHistory = [];

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

const uploadBtn = document.getElementById("upload-btn");
const pdfInput = document.getElementById("pdf-input");
const ingestStatus = document.getElementById("ingest-status");


function addMessage(text, sender) {

    const msgDiv = document.createElement("div");

    msgDiv.classList.add("message");
    msgDiv.classList.add(
        sender === "user" ? "user-message" : "ai-message"
    );

    if (sender === "ai") {

        // Render Markdown and Math Equations for AI responses
        msgDiv.innerHTML = marked.parse(text);

        if (window.MathJax) {
            MathJax.typesetPromise([msgDiv]);
        }

    } else {

        // Keep standard text for user inputs
        msgDiv.textContent = text;

    }

    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}


async function sendQuery() {

    const question = userInput.value.trim();

    if (!question) return;

    // Display user's question
    addMessage(question, "user");

    userInput.value = "";


    // Show loading message
    const loadingId = "load-" + Date.now();

    const loadingDiv = document.createElement("div");

    loadingDiv.classList.add("message", "ai-message");
    loadingDiv.id = loadingId;

    loadingDiv.innerHTML = "<em>Thinking...</em>";

    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;


    try {

        const response = await fetch(`${API_URL}/query`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                question: question,

                chat_history: chatHistory

            })

        });


        const data = await response.json();


        // Remove loading message
        document.getElementById(loadingId)?.remove();


        if (response.ok) {

            // Display AI response
            addMessage(data.answer, "ai");


            // Store this conversation in history
            chatHistory.push({
                role: "user",
                content: question
            });

            chatHistory.push({
                role: "assistant",
                content: data.answer
            });


        } else {

            addMessage(
                "Error: " + (data.detail || "Unable to retrieve answer"),
                "ai"
            );

        }


    } catch (error) {

        document.getElementById(loadingId)?.remove();

        addMessage(
            "Connection Refused! Ensure FastAPI is running on port 8000.",
            "ai"
        );

    }

}


/*
    PDF UPLOAD
*/

uploadBtn.addEventListener("click", async () => {

    const file = pdfInput.files[0];


    // Check whether a file was selected
    if (!file) {

        ingestStatus.textContent =
            "❌ Please select a PDF first.";

        return;
    }


    // Check file type
    if (!file.name.toLowerCase().endsWith(".pdf")) {

        ingestStatus.textContent =
            "❌ Only PDF files are allowed.";

        return;
    }


    uploadBtn.disabled = true;

    ingestStatus.textContent =
        "Uploading and processing...";

    ingestStatus.style.color =
        "var(--text-muted)";


    // Create multipart form data
    const formData = new FormData();

    formData.append("file", file);


    try {

        const response = await fetch(
            `${API_URL}/upload_pdf`,
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (response.ok) {

            ingestStatus.textContent =
                `✅ ${data.message} (${data.chunks_added} chunks added)`;

            ingestStatus.style.color =
                "#00b894";


            // Clear selected file
            pdfInput.value = "";


        } else {

            ingestStatus.textContent =
                "❌ " + (data.detail || "Upload failed.");

            ingestStatus.style.color =
                "#d63031";

        }


    } catch (error) {

        ingestStatus.textContent =
            "❌ Cannot connect to backend.";

        ingestStatus.style.color =
            "#d63031";

    }


    uploadBtn.disabled = false;

});


/*
    SEND BUTTON
*/

sendBtn.addEventListener("click", sendQuery);


/*
    ENTER KEY
*/

userInput.addEventListener("keypress", (e) => {

    if (e.key === "Enter") {

        sendQuery();

    }

});