#!/usr/bin/python3

import os
import sys
import time
import datetime

from datasets import Dataset

from ragas.metrics import (
    answer_correctness,
    answer_similarity,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas import evaluate

from openai import OpenAI


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]
OPENAI_ASSISTANT_ID = os.environ["OPENAI_ASSISTANT_ID"]


questions = [
"Wie lange habe ich Zeit, um mich für die Krankenpflege versichern zu lassen?",

"Wie lange habe ich Zeit, um eine Krankenversicherung abzuschliessen?",

"Wie kann ich mich von der Krankenversicherungspflicht befreien?",
]

ground_truths = [
"Gemäss der Verordnung über die Leistungen in der obligatorischen Krankenpflegeversicherung (KLV) müssen sich Personen innerhalb von drei Monaten nach Wohnsitznahme oder Geburt in der Schweiz für die Krankenpflegeversicherung anmelden.",

"Gemäss der Verordnung über die Krankenversicherung (KVV) müssen sich Personen innerhalb von drei Monaten nach Wohnsitznahme oder Geburt in der Schweiz krankenversichern.",

"""Um sich von der Krankenversicherungspflicht befreien zu lassen, müssen bestimmte Voraussetzungen erfüllt sein. Hier sind einige der wichtigsten Punkte:

1. Personen mit ausländischer Privatversicherung:
   - Eine Befreiung ist möglich, wenn die Unterstellung unter die schweizerische Krankenversicherung eine klare Verschlechterung des bisherigen Versicherungsschutzes oder der bisherigen Kostendeckung zur Folge hätte.
   - Der Versicherungsschutz der Privatversicherung muss deutlich besser sein als der der schweizerischen Krankenversicherung nach KVG.
   - Aufgrund des Alters (mindestens 55 Jahre) und/oder des Gesundheitszustandes ist eine Zusatzversicherung nach VVG bei einem schweizerischen Krankenversicherer nicht oder nur zu kaum tragbaren Bedingungen möglich.

2. Personen mit Aufenthalt ohne Erwerbstätigkeit:
   - Personen mit einer Aufenthaltsbewilligung für Personen ohne Erwerbstätigkeit nach dem Freizügigkeitsabkommen oder dem EFTA-Abkommen können sich befreien lassen, sofern sie während der gesamten Geltungsdauer der Befreiung für Behandlungen in der Schweiz über einen gleichwertigen Versicherungsschutz verfügen.
   - Diese Befreiungsmöglichkeit gilt nicht für Nichterwerbstätige aus der EU/EFTA, die dort im Rahmen eines gesetzlichen Krankenversicherungssystems versichert sind.

3. Personen mit Doppelbelastung:
   - Personen, die nach dem Recht eines anderen Staates obligatorisch krankenversichert sind und für die der Einbezug in die schweizerische Versicherung eine Doppelbelastung bedeuten würde, können sich befreien lassen, sofern sie für Behandlungen in der Schweiz über einen gleichwertigen Versicherungsschutz verfügen.
   - Dieser Befreiungsgrund ist für Personen aus den EU-/EFTA-Staaten nicht anwendbar.

Für die Beantragung der Befreiung sind in der Regel folgende Dokumente erforderlich:
- Aufenthaltsbewilligung (z.B. Kurzaufenthaltsbewilligung L oder Aufenthaltsbewilligung B)
- Aktueller Versicherungsnachweis der zuständigen ausländischen Stelle über den Versicherungsschutz bei Behandlungen in der Schweiz
- Schriftliches Gesuch um Befreiung von der Versicherungspflicht.

Es ist wichtig, die spezifischen Anforderungen und Bedingungen zu prüfen, die für Ihre individuelle Situation gelten.
""",
]


def query_assistant(client, assistant, query):
    print(f"\n\nQUESTION: {query}\n")
    sys.stdout.flush()
    thread = client.beta.threads.create()
    user_message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content=query
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        #instructions="Please address the user as Jane Doe. The user has a premium account."
    )

    # get response
    # see https://platform.openai.com/docs/assistants/tools/file-search/quickstart?context=without-streaming
    messages = list(client.beta.threads.messages.list(
        thread_id=thread.id,
        run_id=run.id
    ))
    if(len(messages) < 1):
        print("no response message")
        print(run)
        sys.stdout.flush()
        return { "content": "", "citations": "" }
        
    message_content = messages[0].content[0].text
    citations = []
    for index, annotation in enumerate(message_content.annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]", 1)
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")
    
    print(message_content.value)
    print("\n".join(citations))
    sys.stdout.flush()

    return {
        "content": message_content.value,
        "citations": "\n".join(citations),
    }

def run_tests():
    # setup assistant
    client = OpenAI()
    assistant = client.beta.assistants.retrieve(OPENAI_ASSISTANT_ID)
    print(f"\nASSISTANT\nid: {assistant.id}\nmodel: {assistant.model}\ntemperature: {assistant.temperature}\ntop_p: {assistant.top_p}\ninstructions:\n{assistant.instructions}")
    sys.stdout.flush()

    # get answers
    answers = []
    citations = []
    for q in questions:
        result = query_assistant(client, assistant, q)
        answers.append(result["content"])
        citations.append(result["citations"])

    # evaluate results
    eval_dataset = Dataset.from_dict({
        "question" : questions,
        "answer" : answers,
        "citations" : citations,
        "ground_truth" : ground_truths
    })

    print("\n\nevaluate Ragas metrics...")
    sys.stdout.flush()
    result = evaluate(
        eval_dataset,
        metrics=[
            answer_correctness,
            answer_similarity,
            #answer_relevancy,
            #faithfulness,
            #context_precision,
            #context_recall,
        ],
    )

    df = result.to_pandas()
    print(df)
    sys.stdout.flush()

    t = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"/opt/app/results/result_{t}.csv", sep=";", encoding="utf-8", lineterminator="\n")

if __name__ == '__main__':
    run_tests()
