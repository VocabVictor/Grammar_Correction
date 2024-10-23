import json
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import jieba
from sklearn.metrics import precision_score, recall_score, fbeta_score, accuracy_score

class GrammarCorrector:
    def __init__(self, api_key, base_url):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def compute_word_level_metrics(input_texts, corrected_texts):
        y_true = []
        y_pred = []
        for input_text, corrected_text in zip(input_texts, corrected_texts):
            input_words = list(jieba.cut(input_text.strip()))
            corrected_words = list(jieba.cut(corrected_text.strip()))
            min_length = min(len(input_words), len(corrected_words))
            y_true.extend(input_words[:min_length])
            y_pred.extend(corrected_words[:min_length])
        
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f0_5 = fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
        accuracy = accuracy_score(y_true, y_pred)
        
        return precision, recall, f0_5, accuracy

    @staticmethod
    def compute_char_level_metrics(input_texts, corrected_texts):
        y_true = []
        y_pred = []
        for input_text, corrected_text in zip(input_texts, corrected_texts):
            input_chars = input_text.strip()
            corrected_chars = corrected_text.strip()
            min_length = min(len(input_chars), len(corrected_chars))
            y_true.extend(input_chars[:min_length])
            y_pred.extend(corrected_chars[:min_length])
        
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f0_5 = fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
        accuracy = accuracy_score(y_true, y_pred)
        
        return precision, recall, f0_5, accuracy

    async def correct_text_with_openai(self, text):
        max_retries = 3
        retry_delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that corrects grammar."},
                        {"role": "user", "content": f"Correct the grammar in the following text:{text}"}
                    ],
                    max_tokens=100,
                    n=1,
                    temperature=0.5,
                )
                # Remove everything before the first colon and strip whitespace
                if ':' in response.choices[0].message.content:
                    corrected_text = response.choices[0].message.content.split(':', 1)[1].strip()
                else:
                    corrected_text = response.choices[0].message.content.strip()
                # Remove newline characters and any text below them
                corrected_text = corrected_text.split('\n')[0]
                # Remove leading and trailing double quotes (both Chinese and English)
                corrected_text = corrected_text.strip('"')
                corrected_text = corrected_text.strip()
                return corrected_text
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    return text

    async def evaluate_corrections(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        tasks = []
        input_texts = []
        for data in input_data:
            text = data['input']
            tasks.append(self.correct_text_with_openai(text.strip()))
            input_texts.append(data['output'])

        corrected_texts = []
        async for future in tqdm(
            tasks,
            total=len(tasks),
            desc="Processing corrections"
        ):
            corrected_text = await future
            corrected_texts.append(corrected_text)

        output_data = {
            'output': corrected_texts
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        word_precision, word_recall, word_f1, word_accuracy = self.compute_word_level_metrics(input_texts, corrected_texts)
        char_precision, char_recall, char_f1, char_accuracy = self.compute_char_level_metrics(input_texts, corrected_texts)
        
        print(f"Word-level metrics:")
        print(f"  Precision: {word_precision:.2f}")
        print(f"  Recall: {word_recall:.2f}")
        print(f"  F0.5-score: {word_f1:.2f}")
        print(f"  Accuracy: {word_accuracy:.2f}")
        print(f"Character-level metrics:")
        print(f"  Precision: {char_precision:.2f}")
        print(f"  Recall: {char_recall:.2f}")
        print(f"  F0.5-score: {char_f1:.2f}")
        print(f"  Accuracy: {char_accuracy:.2f}")

async def main():
    api_key = "sk-sS6W2FBO8YdwmzfGD0Ff600cEe574dCbA47c9959A234D747"  # Replace with your actual OpenAI API key
    base_url = "https://www.gptapi.us/v1"
    input_file = "pseudo_data/nacgec_dev_instruct_zh.json"  # Replace with your input file path
    output_file = "pseudo_data/nacgec_dev_instruct_zh_corrected.json"  # Replace with your desired output file path
    
    corrector = GrammarCorrector(api_key, base_url)
    await corrector.evaluate_corrections(input_file, output_file)

if __name__ == "__main__":
    asyncio.run(main())