# UI Design Specification
## PulseAI Maternal Health Risk Predictor

**Version:** 1.0  
**Last Updated:** October 30, 2025  
**For:** AI-Assisted UI Generation

---

## 1. Design System

### 1.1 Color Palette

**Primary Colors:**
```
Brand Primary:   #2196F3 (Blue)
Brand Dark:      #1976D2
Brand Light:     #BBDEFB
```

**Risk Level Colors:**
```
Low Risk:        #4CAF50 (Green)
Medium Risk:     #FFC107 (Amber/Yellow)
High Risk:       #F44336 (Red)
```

**UI Colors:**
```
Background:      #FAFAFA (Light Gray)
Card/Surface:    #FFFFFF (White)
Text Primary:    #212121 (Dark Gray)
Text Secondary:  #757575 (Medium Gray)
Border:          #E0E0E0 (Light Gray)
Error:           #D32F2F (Red)
Success:         #388E3C (Green)
```

### 1.2 Typography

**Font Family:**
```
Primary: 'Inter', -apple-system, sans-serif
Monospace: 'Roboto Mono', monospace (for numbers)
```

**Font Sizes:**
```
Heading 1:  32px, Bold (Page title)
Heading 2:  24px, Semibold (Section titles)
Heading 3:  18px, Semibold (Card titles)
Body:       16px, Regular (Main text)
Small:      14px, Regular (Labels, hints)
Tiny:       12px, Regular (Captions)
```

**Line Heights:**
```
Headings: 1.2
Body:     1.5
```

### 1.3 Spacing Scale
```
xs:  4px
sm:  8px
md:  16px
lg:  24px
xl:  32px
xxl: 48px
```

### 1.4 Border Radius
```
Small:  4px (inputs, buttons)
Medium: 8px (cards)
Large:  12px (modals)
Round:  50% (icons, badges)
```

### 1.5 Shadows
```
Card:   0 2px 8px rgba(0,0,0,0.1)
Hover:  0 4px 16px rgba(0,0,0,0.15)
Modal:  0 8px 32px rgba(0,0,0,0.2)
```

---

## 2. Component Specifications

### 2.1 Header Component

**Layout:**
- Fixed top position, full width
- Height: 64px
- Background: #2196F3 (Brand Primary)
- Box shadow: 0 2px 4px rgba(0,0,0,0.1)

**Content:**
```
┌─────────────────────────────────────────────────────┐
│ 🏥 PulseAI | Maternal Health Predictor              │
│                                      [Info] [Help]   │
└─────────────────────────────────────────────────────┘
```

**Elements:**
- Logo icon (🏥 or custom) - 24px, white
- Title text - 20px, bold, white
- Subtitle - 14px, white with 80% opacity
- Info/Help icons - 20px, white, top-right corner
- Horizontal padding: 24px

---

### 2.2 Input Form Component

**Container:**
- Max width: 800px, centered
- Background: white card with shadow
- Padding: 32px
- Border radius: 8px
- Margin top: 24px from header

**Layout Structure:**
```
┌─────────────────────────────────────────────┐
│  Patient Vital Signs                        │
│                                             │
│  [Age Input]        [Systolic BP Input]     │
│  [Diastolic Input]  [Blood Sugar Input]     │
│  [Body Temp Input]  [Heart Rate Input]      │
│                                             │
│  [Predict Risk Button] [Reset Button]       │
└─────────────────────────────────────────────┘
```

**Grid:** 2 columns on desktop, 1 column on mobile (<768px)

**Individual Input Field:**
```
Label:
  - Font: 14px, semibold
  - Color: #212121
  - Margin bottom: 8px
  - Include unit in gray text (e.g., "Age (years)")

Input:
  - Height: 48px
  - Border: 1px solid #E0E0E0
  - Border radius: 4px
  - Padding: 12px 16px
  - Font: 16px (mobile-friendly)
  - Background: white
  - Focus state: border #2196F3, outline none
  - Error state: border #F44336, red shake animation

Helper Text:
  - Font: 12px, regular
  - Color: #757575 (normal) or #F44336 (error)
  - Margin top: 4px
  - Show validation range (e.g., "18-50")

Placeholder:
  - Color: #BDBDBD
  - Text: "Enter [field name]"
```

**Validation States:**
```
Default:  Gray border (#E0E0E0)
Valid:    Green border (#4CAF50) + checkmark icon
Invalid:  Red border (#F44336) + error message
Disabled: Gray background (#F5F5F5), no interaction
```

---

### 2.3 Button Components

**Primary Button (Predict Risk):**
```
Size: 48px height × auto width (min 200px)
Background: #2196F3 → #1976D2 on hover
Text: white, 16px, bold
Border radius: 4px
Padding: 0 32px
Shadow: 0 2px 4px rgba(0,0,0,0.2)
Disabled: #BDBDBD background, no hover effect
Cursor: pointer (enabled), not-allowed (disabled)
Icon: Optional loading spinner when submitting
```

**Secondary Button (Reset):**
```
Same size as primary
Background: transparent
Border: 1px solid #E0E0E0
Text: #212121
Hover: background #F5F5F5
```

**Button Layout:**
- Horizontal alignment, 16px gap
- Responsive: Stack vertically on mobile

---

### 2.4 Result Display Component

**Container:**
```
Width: 800px max, centered
Background: white card
Padding: 32px
Border radius: 8px
Margin top: 24px
Shadow: 0 4px 16px rgba(0,0,0,0.1)
Animation: fade-in + slide-up (300ms ease)
```

**Risk Badge (Top of Card):**
```
┌─────────────────────────────────┐
│  🔴 HIGH RISK                   │ ← Large badge
│  Immediate medical attention    │ ← Description
└─────────────────────────────────┘

Badge Styling:
  - Full width, centered text
  - Height: 80px
  - Border radius: 8px
  - Font: 28px bold (risk level), 16px regular (description)
  - Background + text color based on risk:
    • Low: #E8F5E9 background, #2E7D32 text
    • Medium: #FFF8E1 background, #F57C00 text
    • High: #FFEBEE background, #C62828 text
  - Icon: 32px, aligned left
  - Margin bottom: 24px
```

**Confidence Score:**
```
Display: Horizontal bar chart
Label: "Model Confidence: 94%"
Bar:
  - Width: 100% container
  - Height: 12px
  - Background: #E0E0E0
  - Fill: #4CAF50 (width = confidence %)
  - Border radius: 6px
  - Animated fill on load (500ms)
```

**Input Summary Table:**
```
┌──────────────────┬──────────┐
│ Age              │ 35 years │
│ Systolic BP      │ 85 mmHg  │
│ Diastolic BP     │ 60 mmHg  │
│ Blood Sugar      │ 11.0     │
│ Body Temperature │ 102°F    │
│ Heart Rate       │ 86 bpm   │
└──────────────────┴──────────┘

Table Styling:
  - Borders: 1px solid #E0E0E0
  - Padding: 12px per cell
  - Font: 14px
  - Zebra striping: alternating row backgrounds (#FAFAFA)
```

**Action Buttons (Bottom):**
```
[📄 Print Result] [↻ New Assessment] [💾 Save (Optional)]

Styling: Secondary button style
Layout: Horizontal, 12px gap, center-aligned
```

---

### 2.5 Model Info Panel

**Position:** Collapsible panel below header or sidebar

**Collapsed State:**
```
┌─────────────────────────────────┐
│ ℹ️ Model Info [▼]              │
└─────────────────────────────────┘
```

**Expanded State:**
```
┌─────────────────────────────────┐
│ ℹ️ Model Info [▲]              │
├─────────────────────────────────┤
│ Model: Gradient Boosting        │
│ Accuracy: 86.7%                 │
│ High-Risk Recall: 94.5%         │
│ False Negatives: 3/55           │
│ Last Updated: Oct 30, 2025      │
│ [View Documentation →]          │
└─────────────────────────────────┘

Styling:
  - Background: #E3F2FD (light blue)
  - Border: 1px solid #2196F3
  - Padding: 16px
  - Font: 14px
  - Metrics in monospace font
  - Link: #2196F3, underline on hover
```

---

### 2.6 Loading State

**Full-page Loader (Initial Load):**
```
Centered spinner: 48px, #2196F3
Text below: "Loading PulseAI..." (16px, gray)
Background: #FAFAFA
```

**Inline Loader (Prediction in Progress):**
```
Replace button text with:
  [●●●] "Analyzing..."
Spinner: 20px, white (inside button)
Disable button, show loading cursor
```

---

### 2.7 Error States

**Network Error:**
```
┌─────────────────────────────────┐
│ ⚠️ Connection Error             │
│ Unable to reach prediction      │
│ service. Please try again.      │
│ [Retry]                         │
└─────────────────────────────────┘

Styling:
  - Background: #FFF3E0 (light orange)
  - Border: 1px solid #FF9800
  - Icon: 24px, #FF9800
  - Padding: 16px
  - Border radius: 8px
```

**Validation Error (Inline):**
```
Input field:
  - Red border (#F44336)
  - Shake animation (3 shakes, 300ms)
Error text below:
  - Color: #F44336
  - Icon: ⚠️ 12px
  - Text: "Age must be between 18-50"
```

---

### 2.8 Empty State (No Results Yet)

**Center of Screen:**
```
┌─────────────────────────────────┐
│        📋                       │
│   Enter patient vitals above   │
│   to predict health risk level │
└─────────────────────────────────┘

Styling:
  - Icon: 64px, #BDBDBD
  - Text: 16px, #757575, centered
  - Background: transparent
```

---

## 3. Responsive Breakpoints

```css
/* Mobile */
@media (max-width: 767px) {
  - Single column layout
  - Full-width cards (no side margins)
  - Smaller heading sizes (-20%)
  - Stack buttons vertically
  - Input height: 56px (larger touch target)
  - Bottom padding: 80px (avoid keyboard overlap)
}

/* Tablet */
@media (768px - 1023px) {
  - 2-column grid for inputs
  - Side padding: 24px
  - Max card width: 90vw
}

/* Desktop */
@media (1024px+) {
  - Max card width: 800px, centered
  - Side-by-side buttons
  - Hover effects enabled
  - Tooltips on icon hover
}
```

---

## 4. Interactions & Animations

### 4.1 Form Interactions
```
Input Focus:
  - Border color transition (200ms)
  - Label color change to primary
  - Subtle box-shadow glow

Input Change:
  - Real-time validation (debounce 300ms)
  - Instant visual feedback (checkmark/error)

Button Hover:
  - Background darken (150ms)
  - Slight scale up (transform: scale(1.02))
  - Cursor: pointer
```

### 4.2 Result Animations
```
Card Entry:
  - Fade in: opacity 0 → 1 (300ms)
  - Slide up: translateY(20px) → 0 (300ms)
  - Ease-out timing function

Confidence Bar:
  - Fill animation: width 0 → X% (500ms)
  - Ease-in-out timing

Risk Badge:
  - Pulse animation once on appear (1 cycle)
```

### 4.3 Loading Animations
```
Spinner: Continuous rotate (360deg, 1s, linear)
Button Pulse: Opacity 0.8 ↔ 1 (1s, infinite)
```

---

## 5. Accessibility Requirements

### 5.1 Keyboard Navigation
```
Tab Order:
  1. Input fields (top to bottom, left to right)
  2. Predict button
  3. Reset button
  4. Info panel toggle
  5. Action buttons (print, new, save)

Focus Indicators:
  - 2px solid outline (#2196F3)
  - 2px offset from element
  - Visible on all interactive elements
```

### 5.2 Screen Reader Support
```
Form Labels: 
  - Explicit <label for="id"> associations
  - aria-label for icon buttons
  - aria-describedby for helper text

Live Regions:
  - aria-live="polite" for validation messages
  - aria-live="assertive" for error alerts
  - Role="alert" for critical errors

Alt Text:
  - Descriptive text for all icons
  - "Loading prediction" for spinner
  - "High risk result" for risk badge
```

### 5.3 Color Contrast
```
All text combinations must meet WCAG AA:
  - Normal text: 4.5:1 minimum
  - Large text (18px+): 3:1 minimum
  - UI components: 3:1 minimum

Test combinations:
  ✓ #212121 on #FFFFFF (16.1:1)
  ✓ #FFFFFF on #2196F3 (4.6:1)
  ✓ #C62828 on #FFEBEE (9.3:1)
```

---

## 6. Component Hierarchy (React)

```
App
├── Header
│   ├── Logo
│   ├── Title
│   └── InfoButton
│
├── ModelInfoPanel (collapsible)
│   └── MetricsList
│
├── MainContainer
│   ├── InputForm
│   │   ├── InputField (×6)
│   │   │   ├── Label
│   │   │   ├── Input
│   │   │   └── HelperText
│   │   └── ButtonGroup
│   │       ├── PredictButton
│   │       └── ResetButton
│   │
│   ├── LoadingSpinner (conditional)
│   │
│   ├── ResultCard (conditional)
│   │   ├── RiskBadge
│   │   ├── ConfidenceBar
│   │   ├── InputSummaryTable
│   │   └── ActionButtons
│   │
│   └── EmptyState (conditional)
│
└── ErrorAlert (conditional, overlay)
```

---

## 7. Sample CSS Variables

```css
:root {
  /* Colors */
  --color-primary: #2196F3;
  --color-primary-dark: #1976D2;
  --color-primary-light: #BBDEFB;
  
  --color-risk-low: #4CAF50;
  --color-risk-medium: #FFC107;
  --color-risk-high: #F44336;
  
  --color-bg: #FAFAFA;
  --color-surface: #FFFFFF;
  --color-text-primary: #212121;
  --color-text-secondary: #757575;
  --color-border: #E0E0E0;
  
  /* Spacing */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
  
  /* Typography */
  --font-family: 'Inter', sans-serif;
  --font-size-h1: 32px;
  --font-size-h2: 24px;
  --font-size-body: 16px;
  --font-size-small: 14px;
  
  /* Borders */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  
  /* Shadows */
  --shadow-card: 0 2px 8px rgba(0,0,0,0.1);
  --shadow-hover: 0 4px 16px rgba(0,0,0,0.15);
}
```

---

## 8. AI Generation Prompt Template

**Use this prompt with AI image/code generators:**

```
Create a modern, medical-grade web interface for a maternal health 
risk prediction tool. The design should be:

LAYOUT:
- Clean, centered card layout (800px max width)
- White cards on light gray background (#FAFAFA)
- Blue header (#2196F3) with logo and title
- 2-column input grid on desktop, 1-column on mobile

INPUTS:
- 6 labeled input fields with validation
- Fields: Age, Systolic BP, Diastolic BP, Blood Sugar, Body Temp, Heart Rate
- Large touch-friendly inputs (48px height)
- Show units next to labels (years, mmHg, mmol/L, °F, bpm)
- Real-time validation with green checkmarks or red error messages

RESULT CARD:
- Large color-coded risk badge at top:
  • Green for Low Risk (#4CAF50)
  • Yellow for Medium Risk (#FFC107)
  • Red for High Risk (#F44336)
- Confidence bar chart below badge (94% filled)
- Summary table of all input values
- Action buttons at bottom (Print, New Assessment)

STYLE:
- Sans-serif font (Inter or similar)
- Rounded corners (8px)
- Subtle shadows on cards
- Professional medical aesthetic
- High contrast for accessibility
- Smooth animations on interactions

COLORS:
- Primary blue: #2196F3
- Backgrounds: white and #FAFAFA
- Text: #212121 (dark) and #757575 (medium)
- Risk colors: green, yellow, red (as above)
```

---

**Approval:**
- [ ] UI/UX Designer
- [ ] Frontend Developer
- [ ] Product Manager

**Next Steps:** Use this spec for AI-generated mockups or direct implementation
