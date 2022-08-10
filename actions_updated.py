# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import numpy as np
import pickle
#
with open('list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

with open('fitted_model.pickle', 'rb') as modelFile:
    model = pickle.load(modelFile)

class ActionPredictDisease(Action):
#
     def name(self) -> Text:
         return "action_predict_symptom"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker.slots,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        x_test =[]
        
        symptoms = tracker.slots.get("symptom")
        for each in symptoms_list: 
            if each in symptoms:
                x_test.append(1)
            else: 
                x_test.append(0)
        x_test = np.asarray(x_test)
        if symptoms is None:
            output = "could't predict the disease"
        else:
            output = "You may be suffering from {}".format(model.predict(x_test))

        dispatcher.utter_message(text=output)

        return []
