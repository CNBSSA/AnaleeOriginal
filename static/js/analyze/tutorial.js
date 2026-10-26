/**
 * tutorial.js
 * 
 * This module manages the interactive tutorial system for the analyze page.
 * Features include:
 * - First-time user tutorial display
 * - Step-by-step tutorial navigation
 * - Feature highlighting
 * - Progress tracking
 * - Tooltips for UI elements
 * - Tutorial state persistence
 */

export class TutorialManager {
    /**
     * Initialize the tutorial manager
     */
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 4;
        this.initializeTutorial();
    }

    /**
     * Initialize the tutorial system
     */
    initializeTutorial() {
        // Tutorial is opt-in only — do not block the Analyze screen on first visit.
        this.initializeTooltips();
        this.setupEventListeners();
    }

    /**
     * Set up event listeners for tutorial controls
     */
    setupEventListeners() {
        // Next step buttons
        document.querySelectorAll('[onclick^="nextStep"]').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const step = parseInt(button.getAttribute('onclick').match(/\d+/)[0]);
                this.nextStep(step);
            });
        });

        // Previous step buttons
        document.querySelectorAll('[onclick^="prevStep"]').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const step = parseInt(button.getAttribute('onclick').match(/\d+/)[0]);
                this.prevStep(step);
            });
        });

        // Dismiss tutorial button
        const dismissButton = document.querySelector('[onclick="dismissTutorial()"]');
        if (dismissButton) {
            dismissButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.dismissTutorial();
            });
        }

        // Finish tutorial button
        const finishButton = document.querySelector('[onclick="finishTutorial()"]');
        if (finishButton) {
            finishButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.finishTutorial();
            });
        }
    }

    /**
     * Initialize Bootstrap tooltips
     */
    initializeTooltips() {
        const tooltipTriggerList = [].slice.call(
            document.querySelectorAll('[data-bs-toggle="tooltip"]')
        );
        tooltipTriggerList.forEach(tooltipTriggerEl => {
            new bootstrap.Tooltip(tooltipTriggerEl, {
                template: `
                    <div class="tooltip" role="tooltip">
                        <div class="tooltip-arrow"></div>
                        <div class="tooltip-inner bg-primary"></div>
                    </div>
                `
            });
        });
    }

    /**
     * Show the tutorial overlay
     */
    showTutorial() {
        const overlay = document.getElementById('tutorialOverlay');
        if (overlay) {
            overlay.style.display = 'block';
            setTimeout(() => {
                overlay.style.opacity = '1';
            }, 50);
        }
    }

    /**
     * Dismiss the tutorial
     */
    dismissTutorial() {
        const overlay = document.getElementById('tutorialOverlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 300);
            localStorage.setItem('analyzeTutorialSeen', 'true');
        }
    }

    /**
     * Move to the next tutorial step
     * @param {number} step - The step number to move to
     */
    nextStep(step) {
        if (step <= this.totalSteps) {
            const currentStep = document.querySelector('.tutorial-step.active');
            const nextStep = document.querySelector(`.tutorial-step[data-step="${step}"]`);

            if (currentStep && nextStep) {
                this.animateStepTransition(currentStep, nextStep);
                this.updateProgressDots(step);
                this.highlightRelevantFeature(step);
                this.currentStep = step;
            }
        }
    }

    /**
     * Move to the previous tutorial step
     * @param {number} step - The step number to move to
     */
    prevStep(step) {
        if (step >= 1) {
            const currentStep = document.querySelector('.tutorial-step.active');
            const prevStep = document.querySelector(`.tutorial-step[data-step="${step}"]`);

            if (currentStep && prevStep) {
                this.animateStepTransition(currentStep, prevStep);
                this.updateProgressDots(step);
                this.highlightRelevantFeature(step);
                this.currentStep = step;
            }
        }
    }

    /**
     * Animate the transition between tutorial steps
     * @param {HTMLElement} currentStep - The current step element
     * @param {HTMLElement} nextStep - The next step element
     */
    animateStepTransition(currentStep, nextStep) {
        currentStep.style.opacity = '0';
        setTimeout(() => {
            currentStep.classList.remove('active');
            nextStep.classList.add('active');
            setTimeout(() => {
                nextStep.style.opacity = '1';
            }, 50);
        }, 300);
    }

    /**
     * Update the progress dots display
     * @param {number} step - The current step number
     */
    updateProgressDots(step) {
        document.querySelectorAll('.progress-dot').forEach(dot => {
            dot.classList.remove('active');
        });
        const activeDot = document.querySelector(`.progress-dot[data-step="${step}"]`);
        if (activeDot) {
            activeDot.classList.add('active');
        }
    }

    /**
     * Highlight relevant features for the current step
     * @param {number} step - The current step number
     */
    highlightRelevantFeature(step) {
        // Remove existing highlights
        document.querySelectorAll('.highlight-element').forEach(el => {
            el.classList.remove('highlight-element');
        });

        // Add highlight based on step
        switch(step) {
            case 2:
                document.querySelectorAll('.explanation-input').forEach(input => {
                    input.classList.add('highlight-element');
                });
                break;
            case 3:
                document.querySelectorAll('.suggest-account-btn').forEach(btn => {
                    btn.classList.add('highlight-element');
                });
                break;
            case 4:
                document.querySelectorAll('.suggest-btn').forEach(btn => {
                    btn.classList.add('highlight-element');
                });
                break;
        }
    }

    /**
     * Complete the tutorial and set up ongoing help features
     */
    finishTutorial() {
        this.dismissTutorial();
        this.setupOngoingHelp();
        this.showCompletionToast();
    }

    /**
     * Set up ongoing help features after tutorial completion
     */
    setupOngoingHelp() {
        // Add helpful tooltips to the main interface
        document.querySelectorAll('.explanation-input').forEach(input => {
            input.setAttribute('data-bs-toggle', 'tooltip');
            input.setAttribute('data-bs-placement', 'top');
            input.setAttribute('title', 'Type your explanation here. Similar transactions will be automatically detected!');
        });

        document.querySelectorAll('.suggest-account-btn').forEach(btn => {
            btn.setAttribute('data-bs-toggle', 'tooltip');
            btn.setAttribute('data-bs-placement', 'left');
            btn.setAttribute('title', 'Get AI-powered account suggestions based on the transaction details');
        });

        // Reinitialize tooltips after adding new ones
        this.initializeTooltips();
    }

    /**
     * Show completion toast message
     */
    showCompletionToast() {
        const toastElement = document.createElement('div');
        toastElement.className = 'toast position-fixed bottom-0 end-0 m-3';
        toastElement.innerHTML = `
            <div class="toast-header">
                <i class="fas fa-check-circle text-success me-2"></i>
                <strong class="me-auto">Tutorial Completed!</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                You're all set to use our smart features. Hover over elements to see helpful tips!
            </div>
        `;

        document.body.appendChild(toastElement);
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
    }
}

export { TutorialManager };
export default TutorialManager;