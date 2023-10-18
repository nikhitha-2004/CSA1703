symptom('Flu').
symptom('Yellowish eyes and skin').
symptom('Dark color urine').
symptom('Pale bowel movement').
symptom('Fatigue').
symptom('Vomitting').
symptom('Fever').
symptom('Pain in joints').
symptom('Weakness').
symptom('Stomach Pain').

treatment('Flu', 'Drink hot water, avoid cold eatables.').
treatment('Yellowish eyes and skin', 'Put eye drops, have healthy sleep, do not strain your eyes.').
treatment('Dark color urine', 'Drink lots of water, juices and eat fruits. Avoid alcohol consumption.').
treatment('Pale bowel movement', 'Drink lots of water and exercise regularly.').
treatment('Fatigue', 'Drink lots of water, juices and eat fruits.').
treatment('Vomitting', 'Drink salt and water.').
treatment('Fever', 'Put hot water cloth on head and take crocin.').
treatment('Pain in Joints', 'Apply pain killer and take crocin.').
treatment('Weakness', 'Drink salt and water, eat fruits.').
treatment('Stomach Pain', 'Avoid outside food and eat fruits.').

input :- dynamic(patient/2),
    repeat,
    symptom(X),
    write('Does the patient have '),
    write(X),
    write('? '),
    read(Y),
    assert(patient(X,Y)),
    \+ not(X='Stomach Pain'),
    not(output).

disease(hemochromatosis) :- 
    patient('Stomach Pain',yes),
    patient('Pain in joints',yes),
    patient('Weakness',yes),
    patient('Dark color urine',yes),
    patient('Yellowish eyes and skin',yes).

disease(hepatitis_c) :-
    not(disease(hemochromatosis)),
    patient('Pain in joints',yes),
    patient('Fever',yes),
    patient('Fatigue',yes),
    patient('Vomitting',yes),
    patient('Pale bowel movement',yes).

disease(hepatitis_b) :-
    not(disease(hemochromatosis)),
    not(disease(hepatitis_c)),
    patient('Pale bowel movement',yes),
    patient('Dark color urine',yes),
    patient('Yellowish eyes and skin',yes).

disease(hepatitis_a) :-
    not(disease(hemochromatosis)),
    not(disease(hepatitis_c)),
    not(disease(hepatitis_b)),
    patient('Flu',yes),
    patient('Yellowish eyes and skin',yes).

disease(jaundice) :-
    not(disease(hemochromatosis)),
    not(disease(hepatitis_c)),
    not(disease(hepatitis_b)),
    not(disease(hepatitis_a)),
    patient('Yellowish eyes and skin',yes).

disease(flu) :-
    not(disease(hemochromatosis)),
    not(disease(hepatitis_c)),
    not(disease(hepatitis_b)),
    not(disease(hepatitis_a)),  
    patient('Flu',yes).

output:-
    nl,
    possible_diseases,
    nl,
    advice.

possible_diseases :- disease(X), write('The patient may suffer from '), write(X),nl.
advice :- symptom(X), patient(X,yes), treatment(X,Y), write(Y),nl, \+ not(X='Stomach Pain').
