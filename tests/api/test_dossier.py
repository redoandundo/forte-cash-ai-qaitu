from fortebank_audit.api_runtime import format_dossier


def test_event_dossier_preserves_grounded_observations_and_uncertainty():
    dossier = format_dossier({
        'decision': 'insufficient_evidence', 'category': 'hand_or_object_transfer',
        'thesis': 'Предмет перекрыт рукой',
        'observations': [{'timestamp_sec': 12.5, 'visible_fact': 'Рука над столом'}],
        'reasoning': 'Нельзя установить перемещение предмета',
        'risk': 'Нужна проверка сотрудником', 'limitations': ['Предмет не виден'],
    })
    assert 'Недостаточно доказательств' in dossier
    assert '12.50 с: Рука над столом' in dossier
    assert 'Ограничения: Предмет не виден' in dossier
    assert 'Нельзя установить перемещение предмета' in dossier
