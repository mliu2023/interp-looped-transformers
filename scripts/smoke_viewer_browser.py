#!/usr/bin/env python3
"""Optional Playwright browser smoke; run in a CPU allocation, not on login.

Playwright is a validation-only dependency. The viewer itself does not need it.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready-file', required=True)
    parser.add_argument('--screenshot', required=True)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    ready = json.loads(Path(args.ready_file).read_text())
    base = f'http://127.0.0.1:{ready["port"]}'
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(base+'/?token='+ready['token'])

        def ready_ui():
            page.locator('#status[data-state="ready"]').wait_for()
            if not page.locator('#error').is_hidden():
                raise AssertionError(page.locator('#error').inner_text())
            if page.locator('#overlays .map canvas').count() != 6:
                raise AssertionError('Missing invocation overlays')

        ready_ui()
        page.select_option('#layer', index=1)
        ready_ui()
        channels = page.locator('#channel option').count()
        page.select_option('#channel', index=min(1, channels-1))
        ready_ui()
        page.select_option('#stream', 'gate_pre')
        ready_ui()
        page.select_option('#invocation', '6')
        page.select_option('#compare', '6')
        page.select_option('#tail', '4')
        page.locator('#top .card').first.click()
        selected = page.locator('#image-select option:checked').inner_text()
        page.locator('#selected-caption').filter(has_text=selected).wait_for()
        page.select_option('#image-select', index=1)
        selected = page.locator('#image-select option:checked').inner_text()
        page.locator('#selected-caption').filter(has_text=selected).wait_for()
        # Exercise the native range control without eval-based predicates; the
        # application's CSP deliberately permits only scripts served by itself.
        page.locator('#opacity').press('ArrowRight')
        page.select_option('#stream', 'product')
        ready_ui()
        page.screenshot(path=args.screenshot, full_page=True)
        result = {'passed': not errors, 'javascript_errors': errors,
                  'overlays': page.locator('#overlays canvas').count(),
                  'gallery_cards': page.locator('.card').count(),
                  'selected_layer': page.locator('#layer').input_value(),
                  'selected_channel': page.locator('#channel').input_value(),
                  'selected_stream': page.locator('#stream').input_value(),
                  'selected_invocation': page.locator('#invocation').input_value(),
                  'screenshot': str(Path(args.screenshot).resolve())}
        browser.close()
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
