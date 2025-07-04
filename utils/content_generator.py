import logging

logger = logging.getLogger(__name__)


class ContentGenerator:
    def __init__(self):
        self.content_database = {
            "quadratic_equations": {
                "elementary": {
                    "CBSE": self._quadratic_elementary_cbse(),
                    "ICSE": self._quadratic_elementary_icse(),
                    "SSC": self._quadratic_elementary_ssc(),
                },
                "middle_school": {
                    "CBSE": self._quadratic_middle_cbse(),
                    "ICSE": self._quadratic_middle_icse(),
                    "SSC": self._quadratic_middle_ssc(),
                },
                "high_school": {
                    "CBSE": self._quadratic_high_cbse(),
                    "ICSE": self._quadratic_high_icse(),
                    "SSC": self._quadratic_high_ssc(),
                },
            },
            "digestive_system": {
                "elementary": {
                    "CBSE": self._digestive_elementary_cbse(),
                    "ICSE": self._digestive_elementary_icse(),
                    "SSC": self._digestive_elementary_ssc(),
                },
                "middle_school": {
                    "CBSE": self._digestive_middle(),
                    "ICSE": self._digestive_middle_icse(),
                    "SSC": self._digestive_middle_ssc(),
                },
                "high_school": {
                    "CBSE": self._digestive_high_cbse(),
                    "ICSE": self._digestive_high_icse(),
                    "SSC": self._digestive_high_ssc(),
                },
            },
        }

    def generate_content(self, topic: str, level: str, board: str):
        """Generate content for a specific topic, level, and board."""
        return self.content_database.get(topic, {}).get(level, {}).get(board, {})

    # Quadratic Equations - Elementary Level
    def _quadratic_elementary_cbse(self):
        """Elementary CBSE content - Focus on patterns and introduction"""
        return {
            "board": "CBSE",
            "difficulty": "elementary",
            "prerequisites": ["Basic multiplication", "Number patterns"],
            "learning_objectives": [
                "Recognize square numbers",
                "Understand patterns in squares",
                "Introduction to algebraic thinking",
            ],
            "sections": [
                {
                    "title": "Square Numbers and Patterns",
                    "content": """Square numbers are special numbers we get by multiplying a number by itself.
                    Examples: 1×1=1, 2×2=4, 3×3=9, 4×4=16, 5×5=25
                    
                    CBSE Pattern Recognition:
                    • Look at the sequence: 1, 4, 9, 16, 25...
                    • Differences between consecutive squares: 3, 5, 7, 9... (odd numbers!)
                    • This is part of NCERT's approach to building algebraic thinking
                    
                    Activity: Create square patterns using dots or blocks
                    Make a 3×3 square, then 4×4, then 5×5. Count the dots!""",
                }
            ],
        }

    def _quadratic_elementary_icse(self):
        """Elementary ICSE content - Emphasis on basics and projects"""
        return {
            "board": "ICSE",
            "difficulty": "elementary",
            "prerequisites": [
                "Multiplication tables up to 20",
                "Basic geometry shapes",
            ],
            "learning_objectives": [
                "Master square numbers thoroughly",
                "Connect squares to geometry",
                "Develop pattern recognition through projects",
            ],
            "sections": [
                {
                    "title": "Square Numbers - A Strong Foundation",
                    "content": """ICSE emphasizes strong basics, so let's master square numbers completely!
                    
                    What are Square Numbers?
                    • Numbers obtained by multiplying a number by itself
                    • First 10 squares: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100
                    • Memorize these as they form the foundation for quadratic equations
                    
                    Project Work (ICSE Special):
                    1. Create a square number chart up to 20×20
                    2. Draw geometric squares to represent each square number
                    3. Find square numbers in real life (tiles, windows, chess board)
                    
                    Language Connection: Write a short essay on "Square Numbers in Nature"
                    This strengthens both math and English skills (ICSE focus)""",
                }
            ],
        }

    def _quadratic_elementary_ssc(self):
        """Elementary SSC Maharashtra content - Regional approach, practical focus"""
        return {
            "board": "SSC",
            "difficulty": "elementary",
            "prerequisites": [
                "गुणाकार सारणी (Multiplication tables)",
                "Basic number operations",
            ],
            "learning_objectives": [
                "वर्ग संख्या ओळखणे (Recognize square numbers)",
                "Understand practical applications",
                "Local context integration",
            ],
            "sections": [
                {
                    "title": "वर्ग संख्या (Square Numbers)",
                    "content": """Understanding square numbers with Maharashtra context:
                    
                    Square numbers in Marathi: वर्ग संख्या
                    Examples: 
                    • 1×1=1 (एक चा वर्ग = एक)
                    • 2×2=4 (दोन चा वर्ग = चार)
                    • 3×3=9 (तीन चा वर्ग = नऊ)
                    
                    Practical Examples from Maharashtra:
                    • Rangoli patterns often use square designs
                    • Floor tiles in homes (9 tiles = 3×3 square)
                    • Carrom board has 64 squares (8×8)
                    
                    Simple Activity: Count the square tiles in your classroom floor
                    How many tiles make a perfect square?""",
                }
            ],
        }
        """Elementary level content for quadratic equations (introduction to patterns)."""
        return {
            "difficulty": "elementary",
            "prerequisites": [
                "Basic multiplication",
                "Understanding patterns",
                "Simple algebra",
            ],
            "learning_objectives": [
                "Recognize number patterns",
                "Understand square numbers",
                "Solve simple pattern problems",
            ],
            "sections": [
                {
                    "title": "Introduction to Square Numbers",
                    "content": """Square numbers are special numbers that we get when we multiply a number by itself. 
                    For example: 1×1=1, 2×2=4, 3×3=9, 4×4=16, 5×5=25. 
                    We call these square numbers because we can arrange objects in a square shape! 
                    If you have 9 marbles, you can arrange them in 3 rows with 3 marbles each - making a perfect square!
                    
                    Fun fact: The pattern of square numbers grows in a special way. Look at the differences:
                    1 → 4 (difference: 3)
                    4 → 9 (difference: 5)
                    9 → 16 (difference: 7)
                    The differences are odd numbers: 3, 5, 7, 9...""",
                },
                {
                    "title": "Finding Patterns in Numbers",
                    "content": """When we see number patterns, we can predict what comes next!
                    Pattern 1: 1, 4, 9, 16, ? (These are square numbers: 1², 2², 3², 4², so next is 5² = 25)
                    Pattern 2: 2, 8, 18, 32, ? (These are 2 times square numbers: 2×1², 2×2², 2×3², 2×4², so next is 2×5² = 50)
                    
                    We can use these patterns to solve problems:
                    - If a square garden has 4 plants on each side, how many plants total? Answer: 4×4 = 16 plants
                    - If we increase each side to 5 plants, how many total? Answer: 5×5 = 25 plants""",
                },
                {
                    "title": "Simple Square Problems",
                    "content": """Let's solve some fun problems with squares!
                    
                    Problem 1: A checkerboard has 8 squares on each side. How many squares total?
                    Solution: 8 × 8 = 64 squares
                    
                    Problem 2: If you have 36 stickers and want to arrange them in a square, how many on each side?
                    Solution: We need to find what number times itself equals 36. Since 6 × 6 = 36, we put 6 stickers on each side!
                    
                    Problem 3: A square playground needs a fence. If each side is 10 meters, and fence pieces are 1 meter long, how many pieces do we need?
                    Solution: A square has 4 sides, so 10 × 4 = 40 fence pieces""",
                },
            ],
        }

    def _quadratic_middle_cbse(self):
        """Middle school CBSE content - NCERT pattern, competitive exam focus"""
        return {
            "board": "CBSE",
            "difficulty": "middle_school",
            "prerequisites": [
                "Linear equations",
                "Algebraic expressions",
                "Factorization",
            ],
            "learning_objectives": [
                "Solve quadratic equations by factorization",
                "Understand nature of roots",
                "Apply to word problems",
            ],
            "sections": [
                {
                    "title": "Quadratic Equations (NCERT Chapter 4)",
                    "content": """Following NCERT Class 10 syllabus:
                    
                    Standard Form: ax² + bx + c = 0, where a ≠ 0
                    
                    Solving by Factorization:
                    Example: x² + 5x + 6 = 0
                    Step 1: Find factors of 6 that add to 5 → (2,3)
                    Step 2: x² + 2x + 3x + 6 = 0
                    Step 3: x(x + 2) + 3(x + 2) = 0
                    Step 4: (x + 2)(x + 3) = 0
                    Step 5: x = -2 or x = -3
                    
                    CBSE Board Exam Pattern:
                    • 1-2 questions on factorization (2-3 marks each)
                    • Focus on completing the square method
                    • Word problems from NCERT are important""",
                },
                {
                    "title": "Nature of Roots",
                    "content": """For ax² + bx + c = 0, discriminant D = b² - 4ac
                    
                    • If D > 0: Two distinct real roots
                    • If D = 0: Two equal real roots
                    • If D < 0: No real roots
                    
                    This concept is crucial for JEE/NEET preparation later!""",
                },
            ],
        }

    def _quadratic_middle_icse(self):
        """Middle school ICSE content - Detailed approach with more methods"""
        return {
            "board": "ICSE",
            "difficulty": "middle_school",
            "prerequisites": [
                "Strong algebra foundation",
                "Simultaneous equations",
                "Graphs",
            ],
            "learning_objectives": [
                "Master multiple solving methods",
                "Deep understanding of concepts",
                "Project-based learning",
                "Connect to coordinate geometry",
            ],
            "sections": [
                {
                    "title": "Quadratic Equations - Comprehensive Study",
                    "content": """ICSE covers quadratic equations in greater depth:
                    
                    Methods of Solution:
                    1. Factorization Method (detailed)
                    2. Completing the Square (step-by-step)
                    3. Using Formula (derivation required)
                    4. Graphical Method (parabolas)
                    
                    Example with All Methods:
                    Solve: x² - 6x + 5 = 0
                    
                    Method 1 - Factorization:
                    (x - 5)(x - 1) = 0, so x = 5 or x = 1
                    
                    Method 2 - Completing Square:
                    x² - 6x + 9 - 9 + 5 = 0
                    (x - 3)² = 4
                    x - 3 = ±2, so x = 5 or x = 1
                    
                    Method 3 - Formula:
                    x = [6 ± √(36-20)]/2 = [6 ± 4]/2
                    
                    ICSE Project: Graph different quadratic equations and analyze""",
                },
                {
                    "title": "Word Problems (ICSE Emphasis)",
                    "content": """ICSE focuses heavily on application:
                    
                    Problem: A rectangular garden's length is 3m more than its width. 
                    If area = 40 m², find dimensions.
                    
                    Solution with detailed steps:
                    Let width = x meters
                    Length = (x + 3) meters
                    Area = x(x + 3) = 40
                    x² + 3x - 40 = 0
                    
                    Using factorization:
                    (x + 8)(x - 5) = 0
                    x = -8 (rejected) or x = 5
                    
                    Therefore: Width = 5m, Length = 8m
                    
                    Verification is mandatory in ICSE!""",
                },
            ],
        }

    def _quadratic_middle_ssc(self):
        """Middle school SSC Maharashtra content - Practical approach"""
        return {
            "board": "SSC",
            "difficulty": "middle_school",
            "prerequisites": ["बीजगणित (Algebra basics)", "Linear equations"],
            "learning_objectives": [
                "वर्गसमीकरण सोडवणे (Solve quadratic equations)",
                "Practical problem solving",
                "Step-by-step methods as per Balbharati textbook",
            ],
            "sections": [
                {
                    "title": "वर्गसमीकरण (Quadratic Equations)",
                    "content": """Following Maharashtra State Board (Balbharati) textbook:
                    
                    प्रमाणित स्वरूप (Standard Form): ax² + bx + c = 0
                    
                    अवयव पद्धत (Factorization Method):
                    उदाहरण: x² + 7x + 12 = 0
                    
                    Step 1: 12 चे अवयव शोधा जे 7 ची बेरीज देतात
                    12 = 3 × 4 आणि 3 + 4 = 7
                    
                    Step 2: x² + 3x + 4x + 12 = 0
                    Step 3: x(x + 3) + 4(x + 3) = 0
                    Step 4: (x + 3)(x + 4) = 0
                    Step 5: x = -3 किंवा x = -4
                    
                    Maharashtra Board Exam Pattern:
                    • Direct questions from textbook exercises
                    • 2-3 marks for factorization
                    • Activity-based questions""",
                },
                {
                    "title": "दैनंदिन जीवनातील उपयोग (Real-life Applications)",
                    "content": """Practical problems relevant to Maharashtra students:
                    
                    Problem: A farmer in Nashik has rectangular field. 
                    Length = 2× width + 5 meters
                    Area = 1000 sq meters. Find dimensions.
                    
                    Solution:
                    Let width = x meters
                    Length = 2x + 5 meters
                    x(2x + 5) = 1000
                    2x² + 5x - 1000 = 0
                    
                    This connects math to agriculture (important in Maharashtra)""",
                },
            ],
        }
        """Middle school level content for quadratic equations."""
        return {
            "difficulty": "middle_school",
            "prerequisites": [
                "Basic algebra",
                "Solving linear equations",
                "Understanding variables",
                "Graphing on coordinate plane",
            ],
            "learning_objectives": [
                "Understand quadratic expressions",
                "Solve simple quadratic equations by factoring",
                "Graph parabolas",
                "Apply quadratic equations to real problems",
            ],
            "sections": [
                {
                    "title": "What are Quadratic Equations?",
                    "content": """A quadratic equation is an equation where the highest power of the variable is 2.
                    Standard form: ax² + bx + c = 0 (where a ≠ 0)
                    
                    Examples:
                    - x² + 5x + 6 = 0
                    - 2x² - 3x - 2 = 0
                    - x² - 9 = 0
                    
                    The word "quadratic" comes from "quad" meaning square, because the variable is squared.
                    These equations often have two solutions because when we graph them, they form a U-shape (parabola) that can cross the x-axis at two points.""",
                },
                {
                    "title": "Solving by Factoring",
                    "content": """Factoring is one way to solve quadratic equations. We look for two numbers that multiply to give us 'c' and add to give us 'b'.
                    
                    Example 1: Solve x² + 5x + 6 = 0
                    Step 1: Find factors of 6 that add to 5: (2 and 3)
                    Step 2: Write as (x + 2)(x + 3) = 0
                    Step 3: Set each factor to zero: x + 2 = 0 or x + 3 = 0
                    Step 4: Solve: x = -2 or x = -3
                    
                    Example 2: Solve x² - 4 = 0
                    This is a difference of squares: x² - 4 = (x + 2)(x - 2) = 0
                    So x = -2 or x = 2
                    
                    Check your answers by substituting back into the original equation!
                    
                    NOTE: At the middle school level, we focus on factoring method only. 
                    You'll learn other methods like the quadratic formula in high school.""",
                },
                {
                    "title": "Graphing Parabolas",
                    "content": """The graph of a quadratic equation y = ax² + bx + c is called a parabola.
                    
                    Key features:
                    - If a > 0, the parabola opens upward (U-shape)
                    - If a < 0, the parabola opens downward (∩-shape)
                    - The vertex is the highest or lowest point
                    - The axis of symmetry goes through the vertex
                    - The x-intercepts are where the parabola crosses the x-axis (the solutions!)
                    
                    Example: y = x² - 4x + 3
                    - Opens upward (a = 1 > 0)
                    - x-intercepts: Set y = 0 and solve: x = 1 or x = 3
                    - Vertex is at x = 2 (midpoint of x-intercepts)
                    - When x = 2, y = 4 - 8 + 3 = -1, so vertex is (2, -1)""",
                },
                {
                    "title": "Real-World Applications",
                    "content": """Quadratic equations appear in many real situations!
                    
                    1. Projectile Motion: When you throw a ball, its path follows a parabola.
                    Height = -16t² + vt + h (where t = time, v = initial velocity, h = initial height)
                    
                    Example: A ball is thrown upward at 48 ft/s from 6 feet high.
                    Height = -16t² + 48t + 6
                    When does it hit the ground? Set Height = 0 and solve!
                    
                    2. Area Problems: 
                    A rectangular garden has length 3 meters more than its width. If area = 40 m², find dimensions.
                    Let width = x, then length = x + 3
                    Area: x(x + 3) = 40
                    x² + 3x - 40 = 0
                    (x + 8)(x - 5) = 0
                    Since width must be positive, x = 5 meters""",
                },
            ],
        }

    # Quadratic Equations - High School Level
    def _quadratic_high_cbse(self):
        """High school CBSE content - Complete coverage for boards and competitive exams"""
        return {
            "board": "CBSE",
            "difficulty": "high_school",
            "prerequisites": [
                "Complete algebra",
                "Coordinate geometry",
                "Complex numbers basics",
            ],
            "learning_objectives": [
                "Master quadratic formula and discriminant",
                "Understand complex roots",
                "Prepare for JEE/NEET",
                "Advanced problem solving",
            ],
            "sections": [
                {
                    "title": "Quadratic Formula and Discriminant",
                    "content": """CBSE Class 10 Chapter 4 - Complete Coverage:
                    
                    Quadratic Formula: x = [-b ± √(b² - 4ac)] / 2a
                    
                    Discriminant (D) = b² - 4ac
                    • D > 0: Two distinct real roots
                    • D = 0: Equal roots (perfect square)
                    • D < 0: No real roots (complex roots in Class 11)
                    
                    Example for Board Exam:
                    Find nature of roots: 2x² - 7x + 3 = 0
                    D = (-7)² - 4(2)(3) = 49 - 24 = 25 > 0
                    Two distinct real roots
                    
                    Using formula: x = [7 ± 5]/4 = 3 or 1/2
                    
                    JEE/NEET Connection:
                    • Sum of roots = -b/a = 7/2
                    • Product of roots = c/a = 3/2
                    • Verify: 3 + 1/2 = 7/2 ✓""",
                },
                {
                    "title": "Advanced Applications",
                    "content": """CBSE Focus on Competitive Exams:
                    
                    1. Maximum/Minimum Problems:
                    A ball thrown upward: h = -5t² + 30t + 2
                    Find maximum height and time.
                    
                    2. Quadratic Inequalities (intro):
                    When is x² - 5x + 6 > 0?
                    Factor: (x-2)(x-3) > 0
                    Solution: x < 2 or x > 3
                    
                    3. Word Problems (NCERT Based):
                    Speed-Distance-Time problems
                    Work-Time problems
                    Geometric applications""",
                },
            ],
        }

    def _quadratic_high_icse(self):
        """High school ICSE content - Most comprehensive coverage"""
        return {
            "board": "ICSE",
            "difficulty": "high_school",
            "prerequisites": [
                "Advanced algebra",
                "Coordinate geometry",
                "Trigonometry basics",
            ],
            "learning_objectives": [
                "Complete mastery of all methods",
                "Complex problem solving",
                "International standard preparation",
                "Analytical thinking",
            ],
            "sections": [
                {
                    "title": "Comprehensive Quadratic Theory",
                    "content": """ICSE Class 10 - Detailed Coverage:
                    
                    1. Solving Methods (all with proofs):
                    • Factorization (including ac method)
                    • Completing the square (geometrical interpretation)
                    • Quadratic formula (complete derivation)
                    • Graphical method (parabola properties)
                    
                    2. Nature of Roots - Extended:
                    For ax² + bx + c = 0:
                    • Discriminant analysis
                    • Relation between roots and coefficients
                    • Formation of equations with given roots
                    
                    Example (ICSE Level):
                    If α, β are roots of x² - px + q = 0, 
                    find equation whose roots are α² and β²
                    
                    Solution:
                    α + β = p, αβ = q
                    α² + β² = (α + β)² - 2αβ = p² - 2q
                    α²β² = (αβ)² = q²
                    Required equation: x² - (p² - 2q)x + q² = 0""",
                },
                {
                    "title": "Complex Problems and Projects",
                    "content": """ICSE Emphasis on Deep Understanding:
                    
                    1. Simultaneous Quadratic Equations:
                    Solve: x² + y² = 25 and x + y = 7
                    
                    2. Quadratic Functions:
                    • Maxima/Minima without calculus
                    • Graphical transformations
                    • Range and domain concepts
                    
                    3. Project Work (ICSE Special):
                    • Research golden ratio and quadratic equations
                    • Study projectile motion using quadratics
                    • Create a presentation on real-world applications
                    
                    4. Word Problems (Advanced):
                    Complex scenarios involving optimization,
                    economics, and physics applications""",
                },
            ],
        }

    def _quadratic_high_ssc(self):
        """High school SSC Maharashtra content - Board exam focused"""
        return {
            "board": "SSC",
            "difficulty": "high_school",
            "prerequisites": ["Algebra Part I complete", "Geometry basics"],
            "learning_objectives": [
                "Complete SSC board preparation",
                "सूत्र आणि विविध पद्धती (Formula and methods)",
                "Board exam pattern mastery",
            ],
            "sections": [
                {
                    "title": "वर्गसमीकरण - संपूर्ण अभ्यास",
                    "content": """SSC Board Class 10 - Balbharati Textbook:
                    
                    श्रीधराचार्य सूत्र (Quadratic Formula):
                    x = [-b ± √(b² - 4ac)] / 2a
                    
                    विवेचक (Discriminant) = b² - 4ac
                    • b² - 4ac > 0: वेगळी दोन वास्तव मुळे
                    • b² - 4ac = 0: समान मुळे
                    • b² - 4ac < 0: वास्तव मुळे नाहीत
                    
                    Board Exam Important:
                    Practice Set 2.1 to 2.6 thoroughly
                    Problem Set 2 - Must solve all
                    
                    उदाहरण: 3x² + 11x + 10 = 0
                    a = 3, b = 11, c = 10
                    D = 121 - 120 = 1 > 0
                    x = [-11 ± 1]/6 = -10/6 or -12/6
                    x = -5/3 or -2""",
                },
                {
                    "title": "उपयोजन (Applications)",
                    "content": """SSC Board Practical Problems:
                    
                    1. आर्थिक नियोजन (Financial Planning):
                    Compound interest problems using quadratics
                    
                    2. क्षेत्रफळ समस्या (Area Problems):
                    शेतीची जमीन, बाग, इमारत इ.
                    
                    3. वेग-अंतर-काळ (Speed-Distance-Time):
                    रेल्वे, बस यांच्या समस्या
                    
                    Activity Based Question:
                    Make a chart showing different methods
                    to solve x² - 5x + 6 = 0
                    
                    Maharashtra CET Connection:
                    These concepts directly help in 
                    MH-CET entrance exam preparation""",
                },
            ],
        }
        """High school level content for quadratic equations."""
        return {
            "difficulty": "high_school",
            "prerequisites": [
                "Algebra I",
                "Functions",
                "Complex numbers",
                "Completing the square",
            ],
            "learning_objectives": [
                "Master multiple solving methods",
                "Understand the discriminant",
                "Work with complex solutions",
                "Analyze quadratic functions in depth",
            ],
            "sections": [
                {
                    "title": "The Quadratic Formula",
                    "content": """The quadratic formula solves any quadratic equation ax² + bx + c = 0:
                    
                    x = [-b ± √(b² - 4ac)] / 2a
                    
                    Derivation by completing the square:
                    ax² + bx + c = 0
                    x² + (b/a)x + c/a = 0
                    x² + (b/a)x + (b/2a)² = (b/2a)² - c/a
                    (x + b/2a)² = (b² - 4ac) / 4a²
                    x + b/2a = ±√(b² - 4ac) / 2a
                    x = [-b ± √(b² - 4ac)] / 2a
                    
                    Example: 2x² - 7x + 3 = 0
                    a = 2, b = -7, c = 3
                    x = [7 ± √(49 - 24)] / 4 = [7 ± √25] / 4 = [7 ± 5] / 4
                    x = 3 or x = 1/2""",
                },
                {
                    "title": "The Discriminant and Nature of Roots",
                    "content": """The discriminant Δ = b² - 4ac tells us about the solutions:
                    
                    1. If Δ > 0: Two distinct real roots
                       - The parabola crosses the x-axis at two points
                    
                    2. If Δ = 0: One repeated real root (double root)
                       - The parabola touches the x-axis at exactly one point (vertex)
                       - Also called a perfect square trinomial
                    
                    3. If Δ < 0: Two complex conjugate roots
                       - The parabola doesn't cross the x-axis
                       - Roots are of form a ± bi
                    
                    Example: x² - 6x + 10 = 0
                    Δ = 36 - 40 = -4 < 0
                    x = [6 ± √(-4)] / 2 = [6 ± 2i] / 2 = 3 ± i
                    
                    The complex roots always come in conjugate pairs!""",
                },
                {
                    "title": "Advanced Factoring Techniques",
                    "content": """Beyond basic factoring, we have several techniques:
                    
                    1. Factoring by Grouping (for ax² + bx + c where a ≠ 1):
                    6x² + 11x + 3 = 0
                    Find factors of ac = 18 that add to b = 11: (2 and 9)
                    Rewrite: 6x² + 2x + 9x + 3 = 0
                    Group: 2x(3x + 1) + 3(3x + 1) = 0
                    Factor: (2x + 3)(3x + 1) = 0
                    
                    2. Substitution Method:
                    x⁴ - 5x² + 4 = 0
                    Let u = x², then u² - 5u + 4 = 0
                    (u - 4)(u - 1) = 0
                    u = 4 or u = 1
                    x² = 4 or x² = 1
                    x = ±2 or x = ±1
                    
                    3. Rational Root Theorem:
                    For integer coefficients, rational roots are of form p/q where p divides the constant term and q divides the leading coefficient.""",
                },
                {
                    "title": "Applications in Physics and Engineering",
                    "content": """Quadratic equations model many physical phenomena:
                    
                    1. Projectile Motion with Air Resistance:
                    y = y₀ + v₀t - (1/2)gt² where g ≈ 9.8 m/s²
                    
                    Maximum height: t = v₀/g, h_max = y₀ + v₀²/(2g)
                    Range on level ground: R = v₀²sin(2θ)/g
                    
                    2. Optimization Problems:
                    Maximize area with fixed perimeter:
                    Rectangle with perimeter 100m: 2l + 2w = 100
                    Area A = lw = l(50-l) = 50l - l²
                    dA/dl = 50 - 2l = 0, so l = 25m (square maximizes area)
                    
                    3. Economic Applications:
                    Profit = Revenue - Cost
                    If Revenue = -2x² + 100x and Cost = 20x + 500
                    Profit = -2x² + 80x - 500
                    Maximum profit when dP/dx = -4x + 80 = 0, so x = 20 units
                    
                    4. Electronic Circuits:
                    RLC circuits: Resonant frequency when ω² = 1/(LC)
                    Impedance calculations often involve quadratic expressions""",
                },
            ],
        }

    # Digestive System - Elementary Level
    def _digestive_elementary_cbse(self):
        """Elementary CBSE content - NCERT based simple introduction"""
        return {
            "board": "CBSE",
            "difficulty": "elementary",
            "prerequisites": ["Basic body parts", "Healthy eating habits"],
            "learning_objectives": [
                "Identify parts of digestive system",
                "Understand food journey",
                "Learn healthy habits",
            ],
            "sections": [
                {
                    "title": "Our Digestive System",
                    "content": """Following NCERT EVS approach:
                    
                    The Food Journey:
                    1. Mouth - Teeth chew, saliva mixes
                    2. Food pipe (Esophagus) - Food slides down
                    3. Stomach - Like a bag that churns food
                    4. Small intestine - Nutrients absorbed
                    5. Large intestine - Water absorbed
                    6. Waste removed from body
                    
                    CBSE Activity:
                    • Draw and label digestive system
                    • Make a food journey chart
                    • List healthy and unhealthy foods
                    
                    Remember: Chew food 32 times!
                    This helps digestion start in mouth.""",
                }
            ],
        }

    def _digestive_elementary_icse(self):
        """Elementary ICSE content - Detailed with language focus"""
        return {
            "board": "ICSE",
            "difficulty": "elementary",
            "prerequisites": ["Human body basics", "Nutrition awareness"],
            "learning_objectives": [
                "Detailed understanding of digestion",
                "Vocabulary development",
                "Project-based learning",
            ],
            "sections": [
                {
                    "title": "The Digestive System - A Detailed Study",
                    "content": """ICSE's comprehensive approach:
                    
                    Digestive Organs (with functions):
                    1. Mouth - Mechanical & chemical digestion begins
                       • Teeth (incisors, canines, molars)
                       • Tongue - taste and mixing
                       • Saliva - contains enzymes
                    
                    2. Oesophagus - Muscular tube, peristalsis
                    3. Stomach - J-shaped organ, gastric juices
                    4. Small Intestine - 7 meters long! Absorption
                    5. Large Intestine - Water absorption
                    6. Rectum & Anus - Waste elimination
                    
                    ICSE Project:
                    Create a 3D model of digestive system
                    Write an essay: "A Day in Life of Food"
                    
                    New Vocabulary:
                    Digestion, Absorption, Enzymes, Nutrients""",
                }
            ],
        }

    def _digestive_elementary_ssc(self):
        """Elementary SSC Maharashtra content - Local context"""
        return {
            "board": "SSC",
            "difficulty": "elementary",
            "prerequisites": ["शरीराचे भाग (Body parts)", "आहार (Food types)"],
            "learning_objectives": [
                "पचनसंस्था ओळखणे",
                "Traditional food understanding",
                "Local dietary habits",
            ],
            "sections": [
                {
                    "title": "आपली पचनसंस्था (Our Digestive System)",
                    "content": """Maharashtra Board approach with local context:
                    
                    अन्नाचा प्रवास (Food Journey):
                    1. तोंड - दात चावतात, लाळ मिसळते
                    2. अन्ननलिका - अन्न खाली जाते  
                    3. जठर (पोट) - अन्न मिसळते
                    4. लहान आतडे - पोषक द्रव्ये शोषली जातात
                    5. मोठे आतडे - पाणी शोषले जाते
                    
                    महाराष्ट्रीय आहार (Maharashtra Diet):
                    • भाकरी - पचायला सोपी
                    • डाळ - प्रथिने देते
                    • भाजी - जीवनसत्वे
                    • ताक - पचनास मदत
                    
                    Activity: आपल्या घरच्या जेवणाचे चार्ट बनवा""",
                }
            ],
        }
        """Elementary level content for digestive system."""
        return {
            "difficulty": "elementary",
            "prerequisites": ["Basic body parts", "Understanding of eating"],
            "learning_objectives": [
                "Identify main parts of digestive system",
                "Understand the journey of food",
                "Learn healthy eating habits",
            ],
            "sections": [
                {
                    "title": "Your Amazing Digestive System",
                    "content": """Your digestive system is like a long tube that helps your body use the food you eat!
                    
                    The Journey of Food:
                    1. Mouth - Where it all begins! Your teeth chew food into small pieces
                    2. Throat (Esophagus) - A tube that carries food to your stomach
                    3. Stomach - Like a mixing bowl that breaks down food with special juices
                    4. Small Intestine - A long, twisty tube where good nutrients go into your blood
                    5. Large Intestine - Takes out the water and makes waste
                    
                    Fun Fact: If you stretched out your intestines, they would be about 20 feet long - that's taller than 3 grown-ups standing on each other's shoulders!""",
                },
                {
                    "title": "What Happens When You Eat",
                    "content": """Let's follow a bite of apple on its journey!
                    
                    1. CHOMP! Your teeth bite and chew the apple. Your saliva (spit) makes it soft and easy to swallow.
                    
                    2. GULP! When you swallow, the apple goes down your throat. Muscles push it down - you don't even have to think about it!
                    
                    3. SPLASH! The apple lands in your stomach. Your stomach squeezes and mixes it with special juices for 2-4 hours.
                    
                    4. SQUEEZE! The mushy apple moves to the small intestine. Here, the good parts (vitamins and energy) go into your blood to help you grow and play!
                    
                    5. The parts your body doesn't need continue to the large intestine and eventually leave your body when you go to the bathroom.""",
                },
                {
                    "title": "Taking Care of Your Digestive System",
                    "content": """How to keep your tummy happy:
                    
                    1. Chew your food well - Count to 20 while chewing!
                    2. Drink plenty of water - It helps food move through your body
                    3. Eat fruits and vegetables - They have fiber that helps digestion
                    4. Don't eat too fast - Your stomach needs time to tell your brain it's full
                    5. Wash your hands before eating - Keeps bad germs out!
                    
                    Healthy Snack Ideas:
                    - Crunchy carrots with hummus
                    - Apple slices with peanut butter
                    - Yogurt with berries
                    - Whole grain crackers with cheese
                    
                    Remember: Different foods take different times to digest. Fruits digest quickly (30 minutes), while meat takes longer (4-6 hours)!""",
                },
            ],
        }

    def _digestive_middle(self):
        """Middle school level content for digestive system."""
        return {
            "difficulty": "middle_school",
            "prerequisites": [
                "Basic human anatomy",
                "Understanding of nutrients",
                "Chemical vs physical changes",
            ],
            "learning_objectives": [
                "Understand organs and their functions",
                "Learn about enzymes and digestion",
                "Understand nutrient absorption",
                "Connect diet to health",
            ],
            "sections": [
                {
                    "title": "Digestive System Anatomy",
                    "content": """The digestive system consists of the alimentary canal and accessory organs:
                    
                    Main Organs (Alimentary Canal):
                    1. Mouth: Mechanical digestion (teeth) and chemical digestion (salivary amylase)
                    2. Pharynx: Common passage for food and air
                    3. Esophagus: 25cm tube with peristalsis (wave-like muscle contractions)
                    4. Stomach: J-shaped organ that stores food and secretes gastric juice (HCl and pepsin)
                    5. Small Intestine: 6-7 meters long, three parts:
                       - Duodenum: Receives bile and pancreatic juice
                       - Jejunum: Main site of nutrient absorption
                       - Ileum: Absorbs B12 and bile salts
                    6. Large Intestine: 1.5 meters, absorbs water and forms feces
                    
                    Accessory Organs:
                    - Salivary glands: Produce saliva with amylase
                    - Liver: Produces bile for fat digestion
                    - Gallbladder: Stores and concentrates bile
                    - Pancreas: Produces digestive enzymes and bicarbonate""",
                },
                {
                    "title": "Chemical and Mechanical Digestion",
                    "content": """Digestion breaks down food both physically and chemically:
                    
                    Mechanical Digestion:
                    - Chewing in mouth
                    - Churning in stomach
                    - Segmentation in small intestine
                    
                    Chemical Digestion and Enzymes:
                    1. Carbohydrates:
                       - Salivary amylase (mouth): Starch → Maltose
                       - Pancreatic amylase (small intestine): Remaining starch → Maltose
                       - Intestinal enzymes: Disaccharides → Monosaccharides
                    
                    2. Proteins:
                       - Pepsin (stomach): Proteins → Polypeptides
                       - Trypsin (small intestine): Polypeptides → Smaller peptides
                       - Peptidases: Peptides → Amino acids
                    
                    3. Fats:
                       - Bile (from liver): Emulsifies large fat droplets
                       - Pancreatic lipase: Fats → Fatty acids + Glycerol
                    
                    pH Changes: Mouth (6.8) → Stomach (1.5-2) → Small intestine (7-8)""",
                },
                {
                    "title": "Absorption and Transport",
                    "content": """The small intestine is specialized for absorption:
                    
                    Adaptations for Absorption:
                    1. Large surface area (250 m² - size of a tennis court!)
                    2. Villi: Finger-like projections increase surface area
                    3. Microvilli: Tiny projections on villi cells
                    4. Rich blood supply: Capillaries in each villus
                    5. Thin walls: Only one cell thick
                    
                    What Gets Absorbed Where:
                    - Duodenum: Iron, calcium, fats, sugars, water, vitamins A, D, E, K
                    - Jejunum: Sugars, amino acids, fatty acids, vitamins
                    - Ileum: B12, bile salts, remaining nutrients
                    - Large intestine: Water, minerals, vitamins produced by bacteria
                    
                    Transport Methods:
                    - Simple diffusion: Water, fatty acids
                    - Facilitated diffusion: Fructose
                    - Active transport: Glucose, amino acids, minerals
                    - Endocytosis: Large molecules
                    
                    After absorption, nutrients travel via blood to the liver for processing.""",
                },
                {
                    "title": "Digestive Health and Nutrition",
                    "content": """Maintaining digestive health through diet and lifestyle:
                    
                    Essential Nutrients:
                    1. Carbohydrates: 45-65% of calories (whole grains, fruits, vegetables)
                    2. Proteins: 10-35% of calories (lean meats, beans, nuts)
                    3. Fats: 20-35% of calories (focus on unsaturated fats)
                    4. Vitamins: A, B complex, C, D, E, K
                    5. Minerals: Calcium, iron, zinc, magnesium
                    6. Water: 8-10 glasses daily
                    7. Fiber: 25-30g daily (fruits, vegetables, whole grains)
                    
                    Common Digestive Issues:
                    - Acid reflux: Stomach acid backs up into esophagus
                    - Constipation: Often due to low fiber and water intake
                    - Food poisoning: Caused by harmful bacteria
                    - Lactose intolerance: Inability to digest milk sugar
                    
                    Healthy Habits:
                    - Eat slowly and chew thoroughly
                    - Include probiotics (yogurt, fermented foods)
                    - Stay hydrated
                    - Exercise regularly (helps digestion)
                    - Manage stress (affects gut health)""",
                },
            ],
        }

    def _digestive_high(self):
        """High school level content for digestive system."""
        return {
            "difficulty": "high_school",
            "prerequisites": [
                "Cell biology",
                "Biochemistry basics",
                "Enzyme function",
                "Homeostasis",
            ],
            "learning_objectives": [
                "Understand detailed anatomy and histology",
                "Master enzymatic processes and regulation",
                "Learn hormonal control of digestion",
                "Understand diseases and medical applications",
            ],
            "sections": [
                {
                    "title": "Histology and Cellular Organization",
                    "content": """The digestive tract has four main layers:
                    
                    1. Mucosa (innermost):
                       - Epithelium: Simple columnar in stomach/intestines
                       - Lamina propria: Loose connective tissue with immune cells
                       - Muscularis mucosae: Thin smooth muscle layer
                    
                    2. Submucosa:
                       - Dense connective tissue
                       - Contains Meissner's plexus (nerve network)
                       - Blood vessels and lymphatics
                    
                    3. Muscularis externa:
                       - Inner circular and outer longitudinal smooth muscle
                       - Contains Auerbach's plexus (myenteric plexus)
                       - Responsible for peristalsis and segmentation
                    
                    4. Serosa/Adventitia:
                       - Serosa: Visceral peritoneum (most organs)
                       - Adventitia: Fibrous tissue (esophagus, rectum)
                    
                    Specialized Cells:
                    - Parietal cells: Secrete HCl and intrinsic factor
                    - Chief cells: Secrete pepsinogen
                    - G cells: Secrete gastrin
                    - Enterocytes: Absorption and enzyme secretion
                    - Goblet cells: Mucus secretion
                    - Paneth cells: Antimicrobial peptides
                    - Enteroendocrine cells: Hormone secretion""",
                },
                {
                    "title": "Enzymatic Processes and Regulation",
                    "content": """Detailed enzymatic digestion and control mechanisms:
                    
                    Enzyme Activation Cascades:
                    1. Pepsinogen → Pepsin (pH < 2, autocatalytic)
                    2. Trypsinogen → Trypsin (enterokinase)
                    3. Trypsin activates: Chymotrypsinogen, Proelastase, Procarboxypeptidases
                    
                    Enzyme Specificity:
                    - Pepsin: Cleaves at aromatic amino acids (Phe, Trp, Tyr)
                    - Trypsin: Cleaves after basic amino acids (Arg, Lys)
                    - Chymotrypsin: Cleaves after large hydrophobic amino acids
                    - Elastase: Cleaves after small uncharged amino acids
                    - Carboxypeptidase A: Removes aromatic C-terminal amino acids
                    - Carboxypeptidase B: Removes basic C-terminal amino acids
                    
                    Regulation Mechanisms:
                    1. Allosteric regulation: End-product inhibition
                    2. Covalent modification: Phosphorylation of enzymes
                    3. Compartmentalization: Zymogens prevent autodigestion
                    4. pH optimization: Different enzymes work at specific pH ranges
                    5. Cofactor availability: B vitamins, minerals
                    
                    Brush Border Enzymes:
                    - Maltase: Maltose → 2 Glucose
                    - Sucrase: Sucrose → Glucose + Fructose
                    - Lactase: Lactose → Glucose + Galactose
                    - Aminopeptidases: Cleave N-terminal amino acids""",
                },
                {
                    "title": "Hormonal and Neural Control",
                    "content": """Complex regulation of digestive processes:
                    
                    Major Digestive Hormones:
                    1. Gastrin:
                       - Source: G cells (stomach antrum)
                       - Stimuli: Protein, stomach distension, vagal stimulation
                       - Actions: ↑HCl secretion, ↑pepsinogen, ↑gastric motility
                    
                    2. Secretin:
                       - Source: S cells (duodenum)
                       - Stimulus: Acidic chyme (pH < 4.5)
                       - Actions: ↑bicarbonate from pancreas, ↓gastric acid
                    
                    3. Cholecystokinin (CCK):
                       - Source: I cells (duodenum, jejunum)
                       - Stimuli: Fats, proteins
                       - Actions: ↑pancreatic enzymes, gallbladder contraction, ↓gastric emptying
                    
                    4. GIP (Gastric Inhibitory Peptide):
                       - Source: K cells (duodenum, jejunum)
                       - Stimuli: Glucose, fats
                       - Actions: ↑insulin, ↓gastric acid and motility
                    
                    5. Motilin:
                       - Source: M cells (duodenum, jejunum)
                       - Action: Initiates migrating motor complexes (MMCs)
                    
                    Neural Control:
                    - Enteric Nervous System: "Second brain" with 500 million neurons
                    - Parasympathetic (Vagus): ↑secretion, ↑motility
                    - Sympathetic: ↓secretion, ↓motility, vasoconstriction
                    - Local reflexes: Gastrocolic, enterogastric reflexes""",
                },
                {
                    "title": "Clinical Applications and Pathophysiology",
                    "content": """Understanding diseases and medical interventions:
                    
                    Common Gastrointestinal Disorders:
                    1. Peptic Ulcer Disease:
                       - Cause: H. pylori (90%), NSAIDs
                       - Mechanism: Imbalance between protective factors and acid
                       - Treatment: Proton pump inhibitors, antibiotics, H2 blockers
                    
                    2. Inflammatory Bowel Disease:
                       - Crohn's Disease: Transmural inflammation, skip lesions
                       - Ulcerative Colitis: Mucosal inflammation, continuous
                       - Treatment: Immunosuppressants, biologics (anti-TNF)
                    
                    3. Celiac Disease:
                       - Autoimmune response to gluten
                       - Villous atrophy → malabsorption
                       - Diagnosis: Anti-tTG antibodies, biopsy
                    
                    4. Gastroesophageal Reflux Disease (GERD):
                       - Lower esophageal sphincter dysfunction
                       - Complications: Barrett's esophagus, adenocarcinoma
                    
                    Diagnostic Procedures:
                    - Endoscopy: Direct visualization
                    - Barium studies: X-ray with contrast
                    - H. pylori testing: Urea breath test, stool antigen
                    - Manometry: Motility assessment
                    - 24-hour pH monitoring: Acid reflux quantification
                    
                    Absorption Disorders:
                    - Lactose intolerance: Lactase deficiency
                    - Pernicious anemia: Lack of intrinsic factor → B12 deficiency
                    - Steatorrhea: Fat malabsorption (pancreatic insufficiency, bile salt deficiency)""",
                },
            ],
        }

        def _digestive_middle_cbse(self):
            """Middle school CBSE content - NCERT Class 7 Science based"""
            return {
                "board": "CBSE",
                "difficulty": "middle_school",
                "prerequisites": ["Nutrition basics", "Body systems overview"],
                "learning_objectives": [
                    "Understand complete digestive process",
                    "Learn about enzymes and absorption",
                    "Nutrition and balanced diet",
                ],
                "sections": [
                    {
                        "title": "Nutrition in Animals - Digestive System",
                        "content": """NCERT Class 7 Chapter 2:
                    
                    Human Digestive System:
                    1. Buccal Cavity (Mouth):
                       • Teeth - cutting, tearing, grinding
                       • Salivary glands - produce saliva
                       • Tongue - taste and swallowing
                    
                    2. Oesophagus: Food pipe with peristaltic movement
                    
                    3. Stomach:
                       • Secretes HCl and digestive juices
                       • Churns food into semi-liquid (chyme)
                       • Protein digestion begins
                    
                    4. Small Intestine:
                       • Longest part (7.5 meters)
                       • Complete digestion occurs
                       • Villi increase surface area
                       • Absorption of nutrients
                    
                    5. Large Intestine:
                       • Absorbs water
                       • Forms faeces
                    
                    CBSE Focus: Diagram labeling important for exams""",
                    },
                    {
                        "title": "Digestion Process",
                        "content": """Chemical Changes in Food:
                    
                    • Carbohydrates → Simple sugars
                    • Proteins → Amino acids  
                    • Fats → Fatty acids + Glycerol
                    
                    Key Terms for CBSE:
                    - Ingestion: Taking in food
                    - Digestion: Breaking down food
                    - Absorption: Nutrients enter blood
                    - Assimilation: Using nutrients
                    - Egestion: Removing waste""",
                    },
                ],
            }

    def _digestive_middle_icse(self):
        """Middle school ICSE content - Detailed study with terminology"""
        return {
            "board": "ICSE",
            "difficulty": "middle_school",
            "prerequisites": ["Cell structure", "Nutrition types", "Basic chemistry"],
            "learning_objectives": [
                "Master digestive system anatomy",
                "Understand enzymes in detail",
                "Learn disorders and health",
                "Develop scientific vocabulary",
            ],
            "sections": [
                {
                    "title": "The Digestive System - Structure and Function",
                    "content": """ICSE Comprehensive Coverage:
                    
                    Alimentary Canal (9 meters total):
                    
                    1. Mouth (Buccal Cavity):
                       • Teeth types: Incisors (8), Canines (4), 
                         Premolars (8), Molars (12)
                       • Salivary glands: Parotid, Submandibular, Sublingual
                       • Enzyme: Salivary amylase (starch → maltose)
                    
                    2. Pharynx: Common passage for food and air
                    
                    3. Oesophagus: 
                       • 25 cm muscular tube
                       • Peristalsis (wave-like contractions)
                    
                    4. Stomach:
                       • Cardiac sphincter (entry)
                       • Pyloric sphincter (exit)  
                       • Gastric juice: HCl, Pepsin, Mucus
                       • pH 1.5-2.0 (highly acidic)
                    
                    5. Small Intestine (7m):
                       • Duodenum (25cm): Receives bile & pancreatic juice
                       • Jejunum (2.5m): Main absorption site
                       • Ileum (3.5m): B12 absorption""",
                },
                {
                    "title": "Digestive Glands and Enzymes",
                    "content": """ICSE Detail on Chemical Digestion:
                    
                    Major Digestive Glands:
                    
                    1. Liver (largest gland):
                       • Produces bile (no enzymes)
                       • Emulsifies fats
                       • Detoxification
                    
                    2. Pancreas:
                       • Pancreatic amylase: starch → maltose
                       • Trypsin: proteins → peptides
                       • Lipase: fats → fatty acids + glycerol
                    
                    3. Intestinal glands:
                       • Maltase, Sucrase, Lactase
                       • Peptidases
                    
                    ICSE Project: Create enzyme action charts
                    
                    Health Focus:
                    • Importance of fiber
                    • Effects of junk food
                    • Digestive disorders prevention""",
                },
            ],
        }

    def _digestive_middle_ssc(self):
        """Middle school SSC Maharashtra content - State board approach"""
        return {
            "board": "SSC",
            "difficulty": "middle_school",
            "prerequisites": ["मानवी शरीर रचना", "पोषण प्रकार"],
            "learning_objectives": [
                "पचनसंस्थेची रचना व कार्य",
                "पचन प्रक्रिया समजणे",
                "आरोग्यदायी सवयी",
            ],
            "sections": [
                {
                    "title": "मानवी पचनसंस्था",
                    "content": """महाराष्ट्र राज्य मंडळ - इयत्ता 7वी:
                    
                    पचनसंस्थेचे भाग:
                    
                    1. मुखपोकळी:
                       • दात - कृतक, सुळके, चवर्णक
                       • जीभ - चव ओळखणे व गिळणे
                       • लाळग्रंथी - लाळ तयार करते
                    
                    2. अन्ननलिका: 
                       • अन्न खाली नेणारी नळी
                       • क्रमाकुंचन गती
                    
                    3. जठर:
                       • J आकाराचा अवयव
                       • जठररस - HCl, पेप्सिन
                       • अन्न 3-4 तास राहते
                    
                    4. लहान आतडे (7 मीटर):
                       • पचन पूर्ण होते
                       • शोषण होते
                       • रक्तात पोषकद्रव्ये जातात
                    
                    5. मोठे आतडे:
                       • पाणी शोषण
                       • मल तयार होते""",
                },
                {
                    "title": "पचन आणि आरोग्य",
                    "content": """आहार आणि पचन:
                    
                    पचनसंस्थेची काळजी:
                    • वेळेवर जेवण
                    • चांगले चावून खाणे (32 वेळा)
                    • मसालेदार पदार्थ कमी
                    • भरपूर पाणी (8-10 ग्लास)
                    
                    महाराष्ट्रीय आहारातील फायदे:
                    • ज्वारी/बाजरी - फायबर
                    • कडधान्ये - प्रथिने
                    • दही/ताक - प्रोबायोटिक्स
                    • कोथिंबीर/कढीपत्ता - पचनास मदत
                    
                    Activity: आरोग्यदायी आहार तक्ता तयार करा
                    
                    सामान्य समस्या:
                    • अपचन
                    • आम्लता (Acidity)
                    • बद्धकोष्ठता""",
                },
            ],
        }

    # Digestive System - High School Level
    def _digestive_high_cbse(self):
        """High school CBSE content - Class 10 Life Processes"""
        return {
            "board": "CBSE",
            "difficulty": "high_school",
            "prerequisites": ["Cell biology", "Biomolecules", "Enzyme function"],
            "learning_objectives": [
                "Detailed digestive physiology",
                "Enzyme mechanisms",
                "Clinical correlations",
                "Board exam preparation",
            ],
            "sections": [
                {
                    "title": "Life Processes - Nutrition",
                    "content": """NCERT Class 10 Chapter 6:
                    
                    Human Digestive System (Board Level):
                    
                    1. Alimentary Canal:
                    • Mouth → Pharynx → Oesophagus → Stomach → 
                      Small intestine → Large intestine → Anus
                    
                    2. Digestive Glands:
                    • Salivary glands: 3 pairs
                    • Gastric glands: In stomach wall
                    • Liver: Largest gland, bile production
                    • Pancreas: Pancreatic juice
                    
                    Digestion in Different Parts:
                    
                    Mouth:
                    • Mechanical: Teeth mastication
                    • Chemical: Salivary amylase
                    • Starch + Amylase → Maltose (pH 6.8)
                    
                    Stomach:
                    • HCl activates pepsinogen → pepsin
                    • Pepsin: Proteins → Peptones (pH 2)
                    • Mucus protects stomach wall
                    
                    Small Intestine:
                    • Bile emulsifies fats
                    • Pancreatic enzymes complete digestion
                    • Intestinal juice finishes process
                    
                    Important for Boards:
                    • pH values at different parts
                    • Enzyme-substrate specificity
                    • Villi structure and function""",
                },
                {
                    "title": "Absorption and Disorders",
                    "content": """CBSE Board Exam Focus:
                    
                    Absorption Mechanism:
                    • Villi and microvilli increase surface area
                    • Simple sugars, amino acids: Active transport
                    • Fatty acids: Passive diffusion
                    • Water: Osmosis
                    
                    Common Disorders (know for MCQs):
                    • Dental caries: Bacterial action
                    • Peptic ulcer: H. pylori infection
                    • Jaundice: Liver dysfunction
                    • Constipation: Low fiber
                    
                    NEET/Medical entrance connection:
                    These concepts form base for human physiology""",
                },
            ],
        }

    def _digestive_high_icse(self):
        """High school ICSE content - Most comprehensive coverage"""
        return {
            "board": "ICSE",
            "difficulty": "high_school",
            "prerequisites": ["Advanced biology", "Biochemistry", "Histology basics"],
            "learning_objectives": [
                "Master anatomical details",
                "Understand physiological processes",
                "Clinical applications",
                "Research projects",
            ],
            "sections": [
                {
                    "title": "Digestive System - Advanced Study",
                    "content": """ICSE Class 10 - Detailed Coverage:
                    
                    Histology of Digestive Tract:
                    
                    Four Layers (inner to outer):
                    1. Mucosa: 
                       • Epithelium (simple columnar)
                       • Lamina propria
                       • Muscularis mucosae
                    
                    2. Submucosa:
                       • Blood vessels, nerves
                       • Brunner's glands (duodenum)
                    
                    3. Muscularis externa:
                       • Circular layer (inner)
                       • Longitudinal layer (outer)
                       • Auerbach's plexus between
                    
                    4. Serosa: Protective covering
                    
                    Digestive Enzymes (Complete List):
                    
                    Saliva:
                    • α-amylase: Starch → Dextrins + Maltose
                    
                    Gastric Juice:
                    • Pepsin: Proteins → Polypeptides
                    • Gastric lipase: Fats → Fatty acids
                    • Rennin (infants): Milk coagulation
                    
                    Pancreatic Juice:
                    • Trypsin, Chymotrypsin: Proteins
                    • Pancreatic amylase: Starch
                    • Lipase: Fats
                    • Nucleases: Nucleic acids""",
                },
                {
                    "title": "Regulation and Clinical Aspects",
                    "content": """ICSE Advanced Topics:
                    
                    Hormonal Control:
                    • Gastrin: Stimulates HCl secretion
                    • Secretin: Stimulates pancreatic juice
                    • CCK: Bile release, enzyme secretion
                    • GIP: Inhibits gastric activity
                    
                    Neural Control:
                    • Vagus nerve: Parasympathetic
                    • Enteric nervous system
                    • Local reflexes
                    
                    Clinical Correlations:
                    1. Malabsorption syndromes
                    2. Inflammatory bowel disease
                    3. Hepatitis and cirrhosis
                    4. Pancreatic disorders
                    
                    ICSE Project Ideas:
                    • Effect of pH on enzyme activity
                    • Dietary fiber importance study
                    • Traditional vs modern diet analysis
                    • Create working model of peristalsis""",
                },
            ],
        }

    def _digestive_high_ssc(self):
        """High school SSC Maharashtra content - Board focused"""
        return {
            "board": "SSC",
            "difficulty": "high_school",
            "prerequisites": ["जीवशास्त्र मूलतत्वे", "रासायनिक क्रिया"],
            "learning_objectives": [
                "संपूर्ण पचनसंस्था अभ्यास",
                "Board परीक्षा तयारी",
                "आरोग्य जागरूकता",
            ],
            "sections": [
                {
                    "title": "पचनसंस्था - सविस्तर अभ्यास",
                    "content": """SSC Board इयत्ता 10वी:
                    
                    पचनसंस्थेची रचना व कार्ये:
                    
                    1. अन्ननलिका ते गुदद्वार (9 मीटर):
                       • मुख → ग्रसनी → अन्ननलिका → जठर → 
                         लहान आतडे → मोठे आतडे → गुदद्वार
                    
                    2. पाचक ग्रंथी:
                       • लाळग्रंथी: 3 जोड्या
                       • यकृत: सर्वात मोठी ग्रंथी
                       • स्वादुपिंड: पाचक रस
                    
                    विविध भागांत पचन:
                    
                    तोंड (pH 6.8):
                    • यांत्रिक: दात चावणे
                    • रासायनिक: लाळेतील ॲमायलेज
                    
                    जठर (pH 1.5-2):
                    • HCl द्वारे पेप्सिनोजेन सक्रिय
                    • पेप्सिन: प्रथिने → पेप्टोन्स
                    
                    लहान आतडे:
                    • पित्त: चरबीचे इमल्सीकरण
                    • स्वादुपिंडरस: सर्व प्रकारचे पचन
                    
                    Board परीक्षेसाठी महत्वाचे:
                    • pH मूल्ये लक्षात ठेवा
                    • विकृती आणि उपाय""",
                },
                {
                    "title": "आरोग्य आणि पचनसंस्था",
                    "content": """महाराष्ट्र मंडळ - व्यावहारिक ज्ञान:
                    
                    शोषण प्रक्रिया:
                    • रक्तवाहिन्यांमध्ये साखर, अमिनो आम्ले
                    • लसिकावाहिन्यांमध्ये चरबी
                    
                    सामान्य विकृती:
                    • दंतक्षय: जीवाणूंची क्रिया
                    • आम्लपित्त: अति आम्लता
                    • कावीळ: यकृत विकार
                    • मधुमेह: स्वादुपिंड विकार
                    
                    पारंपारिक उपाय:
                    • आवळा: व्हिटामिन C
                    • अदरक: पचनास मदत
                    • जिरे: वायू कमी करते
                    • त्रिफळा: आतड्यांची स्वच्छता
                    
                    Activity: आहार तक्ता आणि BMI""",
                },
            ],
        }