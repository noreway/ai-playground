#!/usr/bin/python3

import os
import datetime

from datasets import Dataset

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
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
"Jede Person mit Wohnsitz in der Schweiz muss sich innerhalb von drei Monaten nach der Wohnsitznahme für die Krankenpflege versichern lassen.",

"Gemäß Art. 3 Abs. 1 KVG musst du dich innerhalb von drei Monaten nach der Wohnsitznahme in der Schweiz für die Krankenpflege versichern lassen.",

"""In der Schweiz gibt es verschiedene Möglichkeiten, sich von der Krankenversicherungspflicht befreien zu lassen, je nach Ihrer persönlichen Situation:

1. **Personen mit ausländischer Privatversicherung**:
   - Sie können sich befreien lassen, wenn:
     - Eine Unterstellung unter die schweizerische Krankenversicherung eine klare Verschlechterung des bisherigen Versicherungsschutzes oder der Kostendeckung zur Folge hat.
     - Aufgrund des Alters (mind. 55 Jahre) oder des Gesundheitszustandes eine Zusatzversicherung in der Schweiz nicht oder nur zu kaum tragbaren Bedingungen möglich ist.

2. **Personen mit Doppelbelastung**:
   - Sie können sich befreien lassen, wenn:
     - Sie nach dem Recht eines anderen Staates obligatorisch krankenversichert sind.
     - Der Einbezug in die schweizerische Versicherung für Sie eine Doppelbelastung bedeutet.
     - Sie für Behandlungen in der Schweiz über einen gleichwertigen Versicherungsschutz verfügen.
   - Hinweis: Diese Befreiung gilt nicht für Personen aus EU-/EFTA-Staaten.

3. **Personen mit Aufenthalt ohne Erwerbstätigkeit**:
   - Sie können sich befreien lassen, wenn:
     - Sie über eine Aufenthaltsbewilligung für Personen ohne Erwerbstätigkeit nach dem Freizügigkeitsabkommen oder dem EFTA-Abkommen verfügen.
     - Sie während der gesamten Geltungsdauer der Befreiung für Behandlungen in der Schweiz über einen gleichwertigen Versicherungsschutz verfügen.
   - Hinweis: Diese Befreiung gilt nicht für Nichterwerbstätige aus der EU/EFTA, die dort im Rahmen eines gesetzlichen Krankenversicherungssystems versichert sind.

### Erforderliche Dokumente für ein Befreiungsgesuch:
- **Für Personen mit ausländischer Privatversicherung**:
  - Schriftliches Gesuch um Befreiung.
  - Aufenthaltsbewilligung (gilt nicht für Schweizer Staatsangehörige).
  - Aktueller Versicherungsnachweis des ausländischen Privatversicherers mit detaillierten Informationen über den Versicherungsschutz.

- **Für Personen mit Doppelbelastung**:
  - Schriftliches Gesuch um Befreiung.
  - Aufenthaltsbewilligung (gilt nicht für Schweizer Staatsangehörige).
  - Aktueller Versicherungsnachweis der zuständigen ausländischen Stelle.

- **Für Personen ohne Erwerbstätigkeit**:
  - Aufenthaltsbewilligung (gilt nicht für Schweizer Staatsangehörige).
  - Ärztlicher Nachweis über den Gesundheitszustand oder Ablehnung eines schweizerischen Krankenversicherers über die Zusatzversicherung.
  - Aktueller Versicherungsnachweis des ausländischen Privatversicherers mit detaillierten Informationen über den Versicherungsschutz.

Falls Sie weitere Informationen oder Klarstellungen benötigen, können Sie sich an die Gemeinsame Einrichtung KVG wenden.""",
]


def query_assistant(client, assistant, query):
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content=query
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        #instructions="Please address the user as Jane Doe. The user has a premium account."
    )
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    
    print(messages)
    return {
        "content": messages.data[0].content[0].text.value,
        "context": [""],
    }

def run_tests():
    # setup assistant
    client = OpenAI()
    assistant = client.beta.assistants.retrieve(OPENAI_ASSISTANT_ID)

    # get answers
    answers = []
    contexts = []
    for q in questions:
        result = query_assistant(client, assistant, q)
        answers.append(result["content"])
        contexts.append([c for c in result["context"]])

    # evaluate results
    eval_dataset = Dataset.from_dict({
        "question" : questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : ground_truths
    })

    result = evaluate(
        eval_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )

    df = result.to_pandas()
    print(df)

    t = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"/opt/app/results/result_{t}.csv", sep=";")

if __name__ == '__main__':
    run_tests()
