\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{technology_watch}[2026/09/26 v2.0 Two-column technology watch article]

% Base class: compact scientific article format
\LoadClass[10pt,a4paper,twocolumn]{article}

% Encoding and typography; documents choose their own language
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{microtype}

% Page layout
\RequirePackage[top=1.8cm,bottom=1.9cm,left=1.8cm,right=1.8cm]{geometry}
\setlength{\columnsep}{0.72cm}
\setlength{\columnseprule}{0pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.38em}
\setlength{\emergencystretch}{1.5em}

% Section headings
\RequirePackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection.}{0.45em}{}
\titlespacing*{\section}{0pt}{1.05em}{0.35em}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection.}{0.4em}{}
\titlespacing*{\subsection}{0pt}{0.8em}{0.25em}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection.}{0.35em}{}
\titlespacing*{\subsubsection}{0pt}{0.65em}{0.15em}

% Compact lists
\RequirePackage{enumitem}
\setlist[itemize]{leftmargin=1.25em,itemsep=0.08em,topsep=0.18em,parsep=0pt}
\setlist[enumerate]{leftmargin=1.45em,itemsep=0.08em,topsep=0.18em,parsep=0pt}

% Scientific elements
\RequirePackage{amsmath,amssymb}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{array}
\RequirePackage{caption}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{fancyhdr}
\RequirePackage{cite}
\RequirePackage{xparse}

\captionsetup{font=small,labelfont=bf,skip=4pt}
\renewcommand{\arraystretch}{1.10}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}

% Restrained running headers and footers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\watchshorttitle}
\fancyhead[R]{\small\watchauthor}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.35pt}
\renewcommand{\footrulewidth}{0pt}
\setlength{\headheight}{13pt}

% Metadata
\newcommand{\watchtitle}{Scientific and technology watch}
\newcommand{\watchshorttitle}{Scientific and technology watch}
\newcommand{\watchsubtitle}{}
\newcommand{\watchauthor}{}
\newcommand{\watchdate}{\today}
\newcommand{\watchcourse}{M2 AIDN}
\newcommand{\watchauthorlabel}{Name:}
\newcommand{\watchdatelabel}{Date:}
\newcommand{\watchabstractlabel}{Abstract --}
\newcommand{\watchkeywordslabel}{Keywords --}
\newcommand{\watchquestionlabel}{Research question --}

\newcommand{\VeilleTitle}[1]{\renewcommand{\watchtitle}{#1}}
\newcommand{\VeilleShortTitle}[1]{\renewcommand{\watchshorttitle}{#1}}
\newcommand{\VeilleSubtitle}[1]{\renewcommand{\watchsubtitle}{#1}}
\newcommand{\VeilleAuthor}[1]{\renewcommand{\watchauthor}{#1}}
\newcommand{\VeilleDate}[1]{\renewcommand{\watchdate}{#1}}
\newcommand{\VeilleCourse}[1]{\renewcommand{\watchcourse}{#1}}

% Full-width title block for the two-column layout.
% Usage: \MakeVeilleTitle{abstract}{keywords}{research question}
\NewDocumentCommand{\MakeVeilleTitle}{m m m}{%
  \twocolumn[{%
    \begin{@twocolumnfalse}
      \noindent\textbf{\watchauthorlabel} \watchauthor\hfill
      \textbf{\watchdatelabel} \watchdate

      \vspace{0.6cm}
      \begin{center}
        {\LARGE\bfseries \watchtitle\par}
        \vspace{0.15cm}
        \ifx\watchsubtitle\empty\else
          {\normalsize\itshape \watchsubtitle\par}
          \vspace{0.12cm}
        \fi
        {\small \watchcourse\par}
      \end{center}
      \vspace{0.42cm}

      \noindent\textbf{\watchabstractlabel} #1
      \vspace{0.18cm}

      \noindent\textbf{\watchkeywordslabel} #2
      \vspace{0.18cm}

      \noindent\textbf{\watchquestionlabel} \textit{#3}
      \vspace{0.55cm}
      \hrule
      \vspace{0.62cm}
    \end{@twocolumnfalse}
  }]
  \thispagestyle{fancy}
}

% Supporting blocks
\newcommand{\QuestionVeille}[1]{\textit{#1}}
\newcommand{\SourceType}[1]{\textbf{#1}}
\newcommand{\weblink}[2]{\href{#1}{#2}}
\newcommand{\doi}[1]{\href{https://doi.org/#1}{doi:#1}}

% Compact source card within the article
% #1 = source, #2 = type, #3 = contribution
\NewDocumentEnvironment{sourceitem}{m m m}{%
  \par\smallskip
  \noindent\textbf{#1}\\[-0.12em]
  {\small\textit{Type : #2.} #3}\par
}{\smallskip}

% Full-width table when needed
\newenvironment{widetable}{\begin{table*}[t]\centering\small}{\end{table*}}

% Compact bibliography
\renewcommand{\refname}{References}
\makeatletter
\renewenvironment{thebibliography}[1]
     {\section*{\refname}%
      \small
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \setlength{\itemsep}{0.28em}
            \setlength{\parsep}{0pt}
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.=\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\makeatother
