#!/usr/bin/env python3
from poc_hardcoding import (
    PermissionLevel,
    main,
)


def get_permission_level(permission_level_str: str) -> PermissionLevel:
    """Convert string to PermissionLevel enum"""
    if permission_level_str.upper() == "PL1":
        return PermissionLevel.PL1_EXECUTE
    elif permission_level_str.upper() == "PL2":
        return PermissionLevel.PL2_WRITE
    else:
        return PermissionLevel.PL0_TEXT_ONLY


if __name__ == "__main__":

    # PERMISSION_LEVEL = "PL0" 
    # permission_level = get_permission_level(PERMISSION_LEVEL)

    for permission_level in [PermissionLevel.PL0_TEXT_ONLY, PermissionLevel.PL1_EXECUTE, PermissionLevel.PL2_WRITE]:
        main(permission_level)

