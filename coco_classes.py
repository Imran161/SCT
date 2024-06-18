SCT_base_classes = [
    {"id": 1, "name": "1", "summable_masks": [1], "subtractive_masks": []},
    {"id": 2, "name": "2", "summable_masks": [2], "subtractive_masks": []},
    {"id": 3, "name": "3", "summable_masks": [3], "subtractive_masks": []},
    {"id": 4, "name": "4", "summable_masks": [4], "subtractive_masks": []},
    {"id": 5, "name": "5", "summable_masks": [5], "subtractive_masks": []},
]


SCT_out_classes = [
    {
        "id": 1,
        "name": "Внутримозговое кровоизлияние",
        "summable_masks": [1],
        "subtractive_masks": [],
    },
    {
        "id": 2,
        "name": "Субарахноидальное кровоизлияние",
        "summable_masks": [2],
        "subtractive_masks": [],
    },
    {
        "id": 3,
        "name": "Cубдуральное кровоизлияние,",
        "summable_masks": [3],
        "subtractive_masks": [],
    },
    {
        "id": 4,
        "name": "Эпидуральное кровоизлияние",
        "summable_masks": [4],
        "subtractive_masks": [],
    },
]


sinusite_base_classes = [
    {"name": "Правая гайморова пазуха (внешний контур)", "id": 1},
    {"name": "Левая гайморова пазуха (внешний контур)", "id": 2},
    {"name": "Левая лобная пазуха (внешний контур)", "id": 3},
    {"name": "Правая лобная пазуха (внешний контур)", "id": 4},
    {"name": "Правая гайморова пазуха (граница внутренней пустоты)", "id": 5},
    {"name": "Левая гайморова пазуха (граница внутренней пустоты)", "id": 6},
    {"name": "Левая лобная пазуха (граница внутренней пустоты)", "id": 7},
    {"name": "Правая лобная пазуха (граница внутренней пустоты)", "id": 8},
    {"name": "Снижение пневматизации околоносовых пазух", "id": 9},
    {"name": "Горизонтальный уровень жидкость-воздух", "id": 10},
    {"name": "Отсутствие пневматизации околоносовых пазух", "id": 11},
    {"name": "Иная патология", "id": 12},
    {"name": "Надпись", "id": 13},
]


sinusite_pat_classes_3 = [
    {
        "name": "Снижение пневматизации околоносовых пазух",
        "id": 1,
        "summable_masks": [9, 11],
        "subtractive_masks": [],
    },
    {
        "name": "Горизонтальный уровень жидкость-воздух",
        "id": 2,
        "summable_masks": [10],
        "subtractive_masks": [],
    },
]
