# Causal Inference Atlas - Interactive Educational Platform

A fully functional, publication-grade educational platform for learning causal inference, featuring interactive code execution, comprehensive content from leading textbooks, and advanced visualization tools.

## 🌟 Key Features

### 1. **Complete Slide-Based Educational Content**
- **35+ Estimands** with detailed explanations
- **100+ Theory Topics** organized by tier
- **Scientifically Valid Content** based on Hernán & Robins "What If" and Pearl's "Causality"
- Complete sections: Data Structure, Diagnostics, Sensitivity, Ethics & Reporting

### 2. **Fully Functional Code Playground**
- **Code Translation Engine**: Automatic translation between R, Python, JavaScript, and HTML
- **Complete Code Execution**: Python (Pyodide), R (WebR), JavaScript, HTML
- **Terminal with Package Manager**: Install packages on-the-fly (`pip install`, `install.packages()`)
- **60+ Lessons** with complete, working code (no placeholders)

### 3. **GitHub Pages Deployment**
- Configured for automatic deployment via GitHub Actions
- Optimized production build with code splitting
- Triggers on push to `main` and `claude/**` branches

## 🚀 Quick Start

### Local Development
```bash
npm install
npm run dev
# Open http://localhost:8080
```

### Production Build
```bash
npm run build
npm run preview
```

## 📚 Content from Leading Textbooks

### From "What If" (Hernán & Robins)
- Chapter 1: Definition of causal effects using potential outcomes
- Chapter 2: Randomization and exchangeability
- Chapter 12: Inverse Probability Weighting (IPW)

### From "Causality" (Pearl)
- Chapter 3: do-calculus and backdoor criterion
- Backdoor adjustment formula implementation
- DAG-based identification strategies

## 🛠️ Technical Stack

- **React 18.3** with TypeScript
- **Vite 5.4** build tool
- **Tailwind CSS 3.4** with custom design system
- **Pyodide 0.26.4**: Python in browser
- **WebR 0.3.3**: R in browser
- **D3.js 7.9**: Interactive visualizations
- **Monaco Editor**: VS Code-like editing experience

## 📊 Statistics

- **35+ Estimands** with complete documentation
- **100+ Theory Topics** across all tiers
- **60+ Lessons** with executable code
- **5 Causal Frameworks** covered
- **18 Study Designs** documented

## 🎯 Using the Platform

1. **Browse estimands** in the library (filterable by tier, framework, design)
2. **Learn theory** through comprehensive slide presentations
3. **Practice coding** in the interactive playground
4. **Translate code** between languages with one click
5. **Visualize relationships** in the network view

## 📖 Documentation

For detailed documentation, see the source code comments and inline help.

## 🙏 Acknowledgments

- Miguel Hernán and Jamie Robins for "What If"
- Judea Pearl for "Causality"
- The broader causal inference community

---

**Built for the causal inference community**
