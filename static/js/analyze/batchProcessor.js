/**
 * batchProcessor.js — phased auto-processing in batches of 10.
 */

import { apiFetch } from './api.js';

const BATCH_SIZE = 10;

export class BatchProcessor {
    constructor(fileId) {
        this.fileId = fileId;
        this.running = false;
        this.offset = 0;
    }

    bindUI() {
        this.progressBar = document.getElementById('batchProgressBar');
        this.progressText = document.getElementById('batchProgressText');
        this.statusText = document.getElementById('batchStatusText');
        this.startButton = document.getElementById('startBatchProcess');
        this.stopButton = document.getElementById('stopBatchProcess');

        this.startButton?.addEventListener('click', () => this.start());
        this.stopButton?.addEventListener('click', () => this.stop());

        // Slice 1 (2026-08-27): the "Analyse the entire statement" hero button
        // drives the SAME loop, with its own whole-file progress + a
        // "review only the exceptions" CTA on completion. Additive — the phased
        // controls above still work independently.
        this.wholeButton = document.getElementById('analyseEntireStatement');
        this.wholeStop = document.getElementById('stopWholeProcess');
        this.wholeBar = document.getElementById('wholeProgressBar');
        this.wholeStatus = document.getElementById('wholeStatusText');
        this.exceptionsLink = document.getElementById('reviewExceptionsLink');
        this.exceptionsCount = document.getElementById('reviewExceptionsCount');
        this.wholeButton?.addEventListener('click', () => this.start());
        this.wholeStop?.addEventListener('click', () => this.stop());
    }

    async start() {
        if (this.running) {
            return;
        }

        this.running = true;
        this.offset = 0;
        this.setUiRunning(true);
        this.updateStatus('Starting batch processing...');

        try {
            let lastResult = null;
            while (this.running) {
                const result = await this.processNextBatch();
                lastResult = result;
                if (!result?.has_more) {
                    this.updateStatus('Batch processing complete. Review suggestions on this page.');
                    break;
                }
                this.offset = result.next_offset;
            }
            this.onComplete(lastResult);
        } catch (error) {
            console.error('Batch processing error:', error);
            this.updateStatus(`Error: ${error.message}`);
        } finally {
            this.running = false;
            this.setUiRunning(false);
        }
    }

    stop() {
        this.running = false;
        this.updateStatus('Batch processing stopped.');
        this.setUiRunning(false);
    }

    async processNextBatch() {
        this.updateStatus(`Processing next ${BATCH_SIZE} transactions...`);

        const result = await apiFetch(`/api/analyze/${this.fileId}/process-batch`, {
            method: 'POST',
            body: JSON.stringify({
                offset: this.offset,
                batch_size: BATCH_SIZE,
            }),
        });

        this.applyBatchResults(result.results || []);
        this.updateProgress(result);

        if (result.processed > 0) {
            this.updateStatus(`Processed ${result.processed} transaction(s). ${result.remaining} remaining.`);
        } else if (result.total_unprocessed === 0) {
            this.updateStatus('All transactions already have accounts or explanations.');
        }

        return result;
    }

    applyBatchResults(results) {
        results.forEach((item) => {
            const select = document.querySelector(`select[name="account_${item.transaction_id}"]`);
            if (!select) {
                return;
            }

            if (item.applied_account_id) {
                select.value = String(item.applied_account_id);
                select.classList.add('border-success');
                return;
            }

            const suggestion = item.suggestion || {};
            if (suggestion.success && suggestion.account) {
                const match = Array.from(select.options).find((option) =>
                    option.text.toLowerCase().includes(String(suggestion.account).toLowerCase())
                );
                if (match) {
                    select.value = match.value;
                    select.classList.add('border-info');
                }
            }
        });
    }

    updateProgress(result) {
        const total = result.total_unprocessed || 0;
        const done = Math.max(0, total - (result.remaining || 0));
        const percent = total > 0 ? Math.round((done / total) * 100) : 100;

        if (this.progressBar) {
            this.progressBar.style.width = `${percent}%`;
            this.progressBar.textContent = `${percent}%`;
        }
        if (this.progressText) {
            this.progressText.textContent = `${done} / ${total} reviewed`;
        }
        // Mirror progress onto the whole-statement hero bar.
        if (this.wholeBar) {
            this.wholeBar.style.width = `${percent}%`;
            this.wholeBar.textContent = `${percent}%`;
        }
        if (this.wholeStatus) {
            this.wholeStatus.textContent = `Analysed ${done} of ${total}…`;
        }
    }

    onComplete(result) {
        // Reveal the "review only the exceptions" CTA. remaining = rows the
        // engine could not confidently place; those are the human's job.
        const remaining = (result && result.remaining) || 0;
        if (this.wholeStatus) {
            this.wholeStatus.textContent = remaining
                ? `Done. ${remaining} transaction${remaining === 1 ? '' : 's'} need your eye.`
                : 'Done — every transaction was placed. Nothing left for you to review.';
        }
        if (this.exceptionsLink && remaining > 0) {
            if (this.exceptionsCount) {
                this.exceptionsCount.textContent =
                    `Review the ${remaining} that need your eye`;
            }
            this.exceptionsLink.classList.remove('d-none');
        }
    }

    updateStatus(message) {
        if (this.statusText) {
            this.statusText.textContent = message;
        }
    }

    setUiRunning(isRunning) {
        if (this.startButton) {
            this.startButton.disabled = isRunning;
        }
        if (this.stopButton) {
            this.stopButton.disabled = !isRunning;
        }
        if (this.wholeButton) {
            this.wholeButton.disabled = isRunning;
        }
        if (this.wholeStop) {
            this.wholeStop.disabled = !isRunning;
        }
    }
}

export default BatchProcessor;
