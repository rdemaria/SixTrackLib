#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import config.SIXTRACKLIB_MODULES

def supports( architecture ):
    return bool( SIXTRACKLIB_MODULES.get( architecture, False ) )

