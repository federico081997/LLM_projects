#!/usr/bin/env python
# coding: utf-8

"""
An Automatic Web Summarizer and Brochure Generator.

This script uses AI models to either:
  1) Summarize the content of a given website.
  2) Identify relevant links on the website for brochure generation and create a short company overview.

The workflow:
  - Prompt the user to select an AI model (either 'gpt-4o-mini' or 'llama3.2').
  - Prompt the user to choose a feature: summarize or create brochure content.
  - Prompt the user to enter a valid, reachable URL.
  - Based on the user’s choices, fetch and parse the page(s), then provide summarizations or compile a brochure in Markdown format.
"""

import json
import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import ollama
from openai import OpenAI
from IPython.display import Markdown, display, update_display

# A list of AI models that this script can interface with.
available_models = ["gpt-4o-mini", "llama3.2"]

# Dictionary mapping integer string keys to feature descriptions.
available_features = {
    "1": "Web Summarizer",
    "2": "Brochure Generator"
}

# Define a set of headers to replicate a typical browser request (often necessary to access certain websites).
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    )
}

class Website:
    """
    The Website class encapsulates methods for fetching a webpage, removing irrelevant content,
    and retrieving its basic textual data, including the title and body. 
    It also extracts all hyperlinks for further processing.
    """
    def __init__(self, url):
        """
        Constructor that initializes and fetches a website’s contents.
        
        :param url: The URL of the webpage to fetch.
        
        Steps:
            1) Send an HTTP GET request to the provided URL using a predefined 'headers' dictionary.
            2) Parse the HTML using BeautifulSoup.
            3) Extract the page title if it exists; otherwise, store a default placeholder.
            4) Remove script, style, img, and input tags to avoid clutter.
            5) Collect textual data by joining the remaining HTML elements with line breaks.
            6) Extract all hyperlinks found on the page and store them in 'self.links'.
        """
        self.url = url
        
        # Send an HTTP GET request to fetch the page content.
        response = requests.get(url, headers=headers)
        
        # Create a BeautifulSoup object to parse and traverse the HTML DOM.
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Retrieve page title if present; store a fallback string if absent.
        if soup.title:
            self.title = soup.title.string
        else:
            self.title = "No title found"
        
        # Check if the webpage actually has a body. If so, remove irrelevant tags.
        if soup.body:
            # Remove scripts, styles, images, and form inputs, which usually do not contain textual content relevant for summarization.
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            # After removing these tags, extract remaining text, preserving paragraph breaks using '\n'.
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            # If there's no body tag, store an empty string to avoid errors in subsequent usage.
            self.text = ""

        # Gather all hyperlinks from <a> tags within the parsed document.
        links = [link.get("href") for link in soup.find_all("a")]
        # Filter out None or empty links and store them in self.links.
        self.links = [link for link in links if link]

    def get_contents(self):
        """
        Presents the webpage’s title and main body text in a simple, readable format.
        
        :return: A string containing the webpage title followed by the parsed body text.
        """
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


def chat_with_model(api_params):
    """
    Interacts with the chosen AI model (OpenAI or Ollama) by passing the necessary parameters.
    
    :param api_params: A dictionary containing the prompts and other metadata required by the model.
    :return: The AI's response (plain text) as a string.
    
    This function switches between:
        1) 'gpt-4o-mini' using the OpenAI client API.
        2) 'llama3.2' using Ollama's chat interface.
    """
    # 'user_model' is set globally in this script (assigned in the main user-interaction section).
    if user_model == "gpt-4o-mini":
        # Load environment variables from a .env file for the OpenAI API key.
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Raise an error if the API key isn't available. This enforces a strict requirement to set up .env.
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY is not set in the environment.")
        
        # Create an instance of the OpenAI class with the retrieved API key.
        openai_client = OpenAI(api_key=api_key)
        
        # The 'chat.completions.create()' method is used to generate a completion from a chat-based model.
        response = openai_client.chat.completions.create(**api_params)
        
        # Return the text content from the first choice in the AI model’s response.
        return response.choices[0].message.content

    elif user_model == "llama3.2":
        # Ollama does not support 'response_format'; remove if present to avoid possible errors.
        if "response_format" in api_params:
            api_params.pop("response_format")
        
        # Pass the parameters to Ollama's 'chat' function. The returned object contains multiple keys; 
        # we focus on 'message' -> 'content' for the textual response.
        response = ollama.chat(**api_params)
        return response["message"]["content"]
    
    else:
        # If an invalid model is selected, raise an error with the list of permissible models.
        raise ValueError(f"❌ Invalid model '{user_model}'. Choose from: {available_models}")


def stream_output():
    """
    Streams the response from the selected AI model incrementally,
    updating the Markdown display in a Jupyter-like environment.
    
    This approach allows partial updates of the UI, rather than 
    waiting for the entire response to finish before display.
    """
    response = ""
    # 'display' is imported from IPython.display to facilitate interactive output updates.
    display_handle = display(Markdown(""), display_id=True)

    # For each 'chunk' in the model's response, append it to the existing response string, 
    # then display the updated text in Markdown format (cleaning up triple backticks in the process).
    for chunk in chat_with_model(api_params):
        response += chunk or ''
        response = response.replace("```", "").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)


def is_valid_url(url):
    """
    Checks if the provided URL string is valid according to a regular expression pattern.
    
    :param url: The URL string to test.
    :return: True if the URL is syntactically valid, otherwise False.
    """
    # Regex pattern checks for optional http/https, then an optional 'www.', followed by 
    # domain name(s), and optional path segments.
    pattern = re.compile(r"^(https?://)?(www\.)?[a-zA-Z-]+(\.[a-zA-Z]{2,})+(/.*)?$")
    return bool(pattern.match(url))


def is_reachable_url(url):
    """
    Determines if the specified URL can be reached by sending an HTTP GET request.
    
    :param url: The URL string to validate for reachability.
    :return: True if the URL returns a 200 OK status code, otherwise False.
    
    Additional notes:
      - If the URL doesn't begin with 'http://' or 'https://', 
        the function prepends 'https://' by default.
      - The 'requests.get()' call includes a timeout to prevent indefinite blocking.
      - A RequestException is caught for generic issues such as timeouts or refused connections.
    """
    # Ensure the URL starts with a valid protocol scheme; otherwise, default to 'https://'.
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        # Attempt to fetch the URL with a specified timeout and allow_redirects.
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        # Return False if any request error is raised (timeout, connection error, etc.).
        return False


def get_all_details(url):
    """
    Gathers textual data from the main page (landing page) and from each link 
    found relevant for the brochure feature. 
    
    :param url: The primary website URL selected by the user.
    :return: A string containing the textual contents of the main page 
             and any additional links deemed relevant for brochure creation.
    
    Steps:
        1) Fetch and parse the main landing page of the URL provided.
        2) Iterate over 'selected_links["links"]', which is expected to be 
           a JSON with objects of the form: {"type": "...", "url": "..."}.
        3) Concatenate each page's content to a single consolidated string.
    """
    # Initialize our result string with a header identifying the landing page.
    result = "Landing page:\n"
    
    # Use the Website class to retrieve the main page’s content.
    result += Website(url).get_contents()
    
    # For each relevant link, add a small title (the "type" field) and fetch its content.
    for link in selected_links["links"]:
        # Denote the type of page to provide context.
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    
    return result


# ------------------------- MAIN CODE BLOCK -------------------------
# Display a friendly welcome message and prompt the user to select an AI model.
print(
    "🚀 WELCOME to the Automatic Web Summarizer & Brochure Generator!\n"
    "Let's start by selecting the AI model you would like to use.\n"
    f"Available models: {', '.join(available_models)}"
)

# Loop until the user enters a valid model name.
while True:
    # Request the user’s model choice and convert to lowercase for consistent comparison.
    user_model = input("👉 Enter your choice: ").lower().strip()
    
    if user_model in available_models:
        print(f"\n✅ {user_model} selected.")
        break
    else:
        print(f"\n❌ Invalid model. Please choose from: {', '.join(available_models)}\n")

# Ask the user to choose a feature: summarization (1) or brochure generation (2).
print(
    "\n🎯 Now select the feature you would like to use:\n"
    "1️⃣ Web Summarizer\n"
    "2️⃣ Brochure Generator"
)

# Loop until a valid feature choice (1 or 2) is entered.
while True:
    feature_selection = input("👉 Enter your choice (1 or 2): ").strip()
    
    if feature_selection in available_features:
        print(f"\n✅ {available_features[feature_selection]} selected.")
        break
    else:
        print("\n❌ Invalid choice. Please enter '1' for Web Summarizer or '2' for Brochure Generator.\n")

# Continuously prompt for a URL and validate it until a valid, reachable URL is entered.
while True:
    url = input("\n🌐 Enter a website URL: ")

    # 1) Check if the URL's syntax is correct.
    if not is_valid_url(url):
        print("\n❌ Invalid format! Make sure to enter a proper URL (e.g., https://example.com).\n")
    # 2) Check if the URL returns a 200 status code.
    elif not is_reachable_url(url):
        print("\n⚠️ The website appears to be unreachable. Please try another URL.\n")
    else:
        # If the protocol scheme is missing, default to 'https://'.
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        print(f"\n✅ The website '{url}' is valid and reachable. 🚀")
        break

# Instantiate the Website class for the main landing page.
website = Website(url)

# ------------------------- FEATURE 1: WEB SUMMARIZER -------------------------
if feature_selection == "1":
    # The system prompt sets the AI's role and guides it to produce a structured summary.
    system_prompt = (
        "You are an assistant that analyzes the contents of a website "
        "and provides a detailed summary, ignoring text that might be navigation-related. "
        "Respond in Markdown format."
    )

    # The user prompt contains the raw content to be summarized, plus instructions on the desired format.
    user_prompt = (
        f"You are analyzing a website titled: **{website.title}**\n\n"
        "### 📌 Website Content Overview:\n"
        f"{website.text}\n\n"
        "🔍 **Task:**\n"
        "- Summarize the website content in Markdown format.\n"
        "- If the website contains **news or announcements**, provide a summary of those as well."
    )

    # The 'messages' structure aligns with OpenAI’s Chat Completion API requirements.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Define parameters for the AI model, including which model is used and the prompt messages.
    api_params = {
        "model": user_model,
        "messages": messages
    }

    # Print status message and invoke the 'stream_output' function to display results incrementally.
    print(f"\n📝 Generating a summary for the website: **{website.title}**.\n")
    stream_output()

# ------------------------- FEATURE 2: BROCHURE GENERATOR -------------------------
elif feature_selection == "2":
    # For the brochure generation feature, we must first retrieve and parse relevant links from the webpage.
    link_system_prompt = (
        "You are given a list of links extracted from a company's webpage. "
        "Your task is to determine which links are most relevant for inclusion in a company brochure.\n\n"
        "⚠️ IMPORTANT: Respond with a valid JSON object ONLY. No extra text, no explanations, no formatting issues.\n\n"
        "Your response MUST follow this JSON structure:\n"
        "```json\n"
        "{\n"
        '    "links": [\n'
        '        {"type": "about page", "url": "https://full.url/goes/here/about"},\n'
        '        {"type": "careers page", "url": "https://another.full.url/careers"}\n'
        "    ]\n"
        "}\n"
        "```"
    )

    # Build the prompt for the AI by listing the discovered links on the main page.
    link_user_prompt = (
        f"Here is the list of links found on the website of {website.url}.\n\n"
        "🔍 **Task:** Identify and return only the relevant links suitable for a company brochure. "
        "Ensure that the response includes the **full https URL** in JSON format.\n\n"
        "**🚫 Do NOT include:**\n"
        "- Terms of Service\n"
        "- Privacy Policy\n"
        "- Email links\n\n"
        "**🔗 Links (some might be relative):**\n"
    )
    link_user_prompt += "\n".join(website.links)

    # Compile messages for the AI. The system prompt clarifies the response requirements,
    # and the user prompt provides actual data (the link list).
    messages = [
        {"role": "system", "content": link_system_prompt},
        {"role": "user", "content": link_user_prompt}
    ]

    # Parameters for the AI call, specifying a JSON response format for link analysis.
    api_params = {
        "model": user_model,
        "messages": messages,
        "response_format": {"type": "json_object"}
    }

    print("\nObtaining relevant links...\n")
    # Make a direct call (without streaming) to retrieve the JSON data from the model.
    selected_links = json.loads(chat_with_model(api_params))

    # Once we have the relevant links, we parse those pages and gather their textual content.
    print("Gathering information from relevant links...\n")

    # The next system prompt explains how to build a brochure from these selected links.
    system_prompt = (
        "You are an assistant that analyzes the contents of several relevant pages from a company website "
        "and creates a short brochure about the company for prospective customers, investors, and recruits. "
        "Respond in Markdown format. Include details of company culture, customers, and careers/jobs if available."
    )

    # Compose a user prompt that includes the landing page content plus each relevant link.
    user_prompt = (
        f"You are looking at a company called: {website.title}\n"
        "Here are the contents of its landing page and other relevant pages; "
        "use this information to build a short brochure of the company in Markdown.\n"
    )
    user_prompt += get_all_details(url)

    # Prepare the final messages for the AI to produce the brochure.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    api_params = {
        "model": user_model,
        "messages": messages
    }

    print(f"\n📝 Generating a brochure for the company: **{website.title}**.\n")
    # Stream the AI's output for a real-time user experience.
    stream_output()




