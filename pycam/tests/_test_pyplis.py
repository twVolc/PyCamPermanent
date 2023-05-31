# -*- coding: utf-8 -*-

import pyplis


class TestPyplis:

    def test_get_source(self):
        """Tests online retrieval of source information"""

        source = pyplis.inout.get_source_info_online('eginnniun')

        print(source)