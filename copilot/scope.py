"""Which visits this user may see - the same rules as the CRM order list.

Copied from orders CustomerOrderV2 (order list) and
OrderDashboardNewController::labUserFilter, so the Copilot never shows a user
an order they could not open in the CRM:

- lab user (role 25):
    lab 1657            -> only visits they created themselves
    has lab locations   -> visits tested at those locations
    otherwise           -> visits assigned to their lab
- role 27               -> every visit of the customer
- facility user         -> visits of their facility
- anyone else           -> every visit of the customer
then, for everyone, the user's city list (user_cities) when they have one.
Always: same customer, not deleted, is_active = 1.
"""
from .db import in_list


def visit_scope(p, alias='v'):
    """(sql, params) to AND onto a query over `visits` aliased as `alias`."""
    a = alias
    clauses = [f"{a}.customer_id = %s", f"{a}.deleted_at IS NULL", f"{a}.is_active = 1"]
    params = [p.customer_id]

    if p.role_id == 25:
        if p.user_lab == '1657':
            clauses.append(f"{a}.created_by = %s")
            params.append(str(p.user_id))
        elif p.lab_location_ids:
            marks, values = in_list(p.lab_location_ids)
            clauses.append(
                f"{a}.id IN (SELECT vtl.visit_id FROM visit_test_labs vtl "
                f"WHERE vtl.deleted_at IS NULL AND vtl.lab_location_id IN ({marks}))"
            )
            params.extend(values)
        else:
            clauses.append(f"{a}.assign_lab_id = %s")
            params.append(p.user_lab)
    elif p.role_id == 27:
        pass
    elif p.is_facility == 1 and p.user_facility:
        clauses.append(f"{a}.facility_id = %s")
        params.append(p.user_facility)

    if p.city_ids:
        marks, values = in_list(p.city_ids)
        clauses.append(f"{a}.city_id IN ({marks})")
        params.extend(values)

    return ' AND '.join(clauses), params
