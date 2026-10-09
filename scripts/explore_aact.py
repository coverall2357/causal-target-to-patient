"""Profile rheumatoid arthritis trials in AACT before writing inclusion filters.

Exploratory: shows how phase, status, and intervention type are actually coded,
so the trial-universe filters can be written against real values.
"""

from causal_target_to_patient.sources.aact import get_engine, query_ra


def main():
    engine = get_engine()

    total = query_ra(engine, "SELECT COUNT(*) AS n FROM ra")
    print(f"RA studies (any type): {total.n.iloc[0]}\n")

    by_type = query_ra(engine, """
        SELECT s.study_type, COUNT(*) AS n
        FROM ra JOIN ctgov.studies s USING (nct_id)
        GROUP BY s.study_type ORDER BY n DESC
    """)
    print("By study type:\n", by_type.to_string(index=False), "\n")

    # The remaining profiles are restricted to interventional studies.
    for field in ["phase", "overall_status"]:
        counts = query_ra(engine, f"""
            SELECT s.{field}, COUNT(*) AS n
            FROM ra JOIN ctgov.studies s USING (nct_id)
            WHERE s.study_type ILIKE 'interventional'
            GROUP BY s.{field} ORDER BY n DESC
        """)
        print(f"Interventional, by {field}:\n", counts.to_string(index=False), "\n")

    by_intervention = query_ra(engine, """
        SELECT i.intervention_type, COUNT(DISTINCT i.nct_id) AS n_studies
        FROM ra JOIN ctgov.interventions i USING (nct_id)
        GROUP BY i.intervention_type ORDER BY n_studies DESC
    """)
    print("By intervention type:\n", by_intervention.to_string(index=False), "\n")

    with_reason = query_ra(engine, """
        SELECT COUNT(*) AS n
        FROM ra JOIN ctgov.studies s USING (nct_id)
        WHERE s.why_stopped IS NOT NULL AND s.why_stopped <> ''
    """)
    print(f"Studies with a stated reason for stopping: {with_reason.n.iloc[0]}")


if __name__ == "__main__":
    main()
