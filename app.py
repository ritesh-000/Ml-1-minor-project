# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# import torch
# import re 
# from fastapi.templating import Jinja2Templates # UI
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles

# app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
# tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# # device
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

# model.to(device)

# templates = Jinja2Templates(directory="templates")

# class DialogueInput(BaseModel):
#     dialogue: str

# def clean_data(text):
#     text = re.sub(r"\r\n", " ", text) # lines
#     text = re.sub(r"\s+", " ", text) # spaces
#     text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
#     text = text.strip().lower()
#     return text

# def summarize_dialogue(dialogue : str) -> str:
#     dialogue = clean_data(dialogue) # clean

#     # tokenize
#     inputs = tokenizer(
#         dialogue,
#         padding="max_length",
#         max_length=512,
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)

#     # generate the summary => token ids
#     model.to(device)
#     targets = model.generate(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_length=150,
#         num_beams=4,
#         early_stopping=True
#     )
    
#     # decoded our output
#     summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
#     return summary


# # API endpoints
# @app.post("/summarize/")
# async def summarize(dialogue_input: DialogueInput):
#     summary = summarize_dialogue(dialogue_input.dialogue)
#     return {"summary": summary}

# # @app.get("/", response_class=HTMLResponse)
# # async def home(request: Request):
# #     return templates.TemplateResponse("index.html", {"request": request})
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse(
#         request=request,
#         name="index.html"
#     )

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# ---------------------------------------------------
# Device Selection
# ---------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ---------------------------------------------------
# Model Configuration
# ---------------------------------------------------
MODEL_NAME = "t5-small"

# Lazy loading variables
model = None
tokenizer = None


def load_model():
    """
    Load the model and tokenizer only when needed.
    This allows Render to start the server quickly
    before downloading/loading the model.
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        print("Loading T5 model...")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")


# ---------------------------------------------------
# Templates
# ---------------------------------------------------
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class DialogueInput(BaseModel):
    dialogue: str


# ---------------------------------------------------
# Text Cleaning
# ---------------------------------------------------
def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text


# ---------------------------------------------------
# Summarization Function
# ---------------------------------------------------
def summarize_dialogue(dialogue: str) -> str:
    # Load model on first request
    load_model()

    # Clean input text
    dialogue = clean_data(dialogue)

    # Add T5 prefix
    input_text = "summarize: " + dialogue

    # Tokenize
    inputs = tokenizer(
        input_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # Move tensors to device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # Decode output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# ---------------------------------------------------
# API Endpoint
# ---------------------------------------------------
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


# ---------------------------------------------------
# Home Page
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ---------------------------------------------------
# Health Check Endpoint (Optional)
# ---------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}