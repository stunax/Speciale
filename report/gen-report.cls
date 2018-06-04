% General report format template with useful utilities and packages.
%
% NOT TESTED ON NEW INSTALLATION, only tested on a sharelatex instance.
% 
% Filename: gen-report.cls
% 
% Features:
%   - Dank animation package
%   - Looks... better than standard article?
%   - Lots of other useful packages, imo.
% 
%   LICENSE: Public domain: You are free to do whatever you want with
%   this template.

% DISCLAIMER STUFF AND FORMALITIES HERE
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{gen-report}[2017/1/01, Mark Wulff]

\typeout{A general report class with useful packages and utilities}
\typeout{- original by Mark Wulff}
\typeout{}
\typeout{Available at... well... I guess you got it somehow to read this?}
\typeout{}

% -----------------------------------------------------------------------------
% GENERAL FOR THE DOCUMENT/VARIABLES WHICH CAN BE CHANGED

% The text displayed in the draft header
\newcommand{\draftheadertext}{\textcolor{red}{DRAFT: \today, \currenttime}}


% -----------------------------------------------------------------------------
% DOCUMENT FORMATTING AND STYLE

% Bases on the article class, not truly kosher if you are writing a long paper 
% but hey I assume you don't give a fuck anyway, right?
\LoadClass[12pt, a4paper]{article}

\usepackage[utf8]{inputenc} % In the north, we like utf8 for our æ,ø and å

\usepackage{lastpage} % Get the last page as a command for the footer

% PAGE FORMATTING!!!!
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{} % Put nothing in all 6 header/footer fields
\renewcommand\headrulewidth{0pt} % Removes the header line thing
\fancyfoot[R]{\thepage \hspace{1pt} of \hspace{1pt}\pageref{LastPage}} % Formats page numbering, lower right corner

% So, if its not a finished paper, a draft text is printed in the header
\ifpaperdraft
  \fancyhead[C]{\draftheadertext}
\else
  % Nothing in the header
\fi

% Defines the page style plain to not have anything other than DRAFT
% You can add more if you acturally want something here from the general page style or different
\fancypagestyle{plain}{
    \renewcommand\headrulewidth{0pt}
    \fancyhf{}
    
    \ifpaperdraft
        \fancyhead[C]{\draftheadertext}
    \else
        % Nothing in the header
    \fi
}


% Reinforce the margins for the a4 paper, can add: ,left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm in the options
% Add option "showframe" to see how the page is divided 
\usepackage[a4paper, headheight=15pt]{geometry}

% IF YOU CHANGE THE MARGIN OF THE PAPER, YOU SHOULD ALSO CHANGE THIS TO SEE YOUR TODOS BETTER!
\setlength{\marginparwidth}{2.5cm} % For the todo notes to have more space

% -----------------------------------------------------------------------------
% PRIOitized TODO NOTES!

% Fix now!!!
\newcommand{\todored}[1] 
{
  \todo[color=red]{#1}
}

% Fix at some point before deadline
\newcommand{\todoyellow}[1] 
{
  \todo[color=yellow]{#1}
}

% Fix if you have the energy at some point, or just leave as is
\newcommand{\todogreen}[1] 
{
  \todo[color=green]{#1}
}


% -----------------------------------------------------------------------------
% ADDITIONAL PACKAGES WHICH ARE USEFUL

% If draft paper is set true, then include todos, if false remove all todos
% Set in the top of main.tex, this is to make it easy to be a lazy fuck so 
% we don't have to spend ages actually fixing our fucking todos...
\ifpaperdraft
  \usepackage{todonotes}
\else
  \usepackage[disable]{todonotes}
\fi

\usepackage{eso-pic} % \AddToShipoutPicture
\usepackage[T1]{fontenc} % For words with accented characters like "ô"
\usepackage{textcomp} % Prodices additional characters like yen (currency)
\usepackage[scaled=.92]{helvet} % for proper fonts
\usepackage{graphicx} % For EPS use the graphics package instead
\usepackage{booktabs} % For pretty table rules
\usepackage{ccicons}  % For Creative Commons citation icons, \cclogo (for example)
\usepackage{ragged2e} % For tighter hyphenation
\usepackage{float} % For using alignment options like h! 
\usepackage{parskip} % So you dont have to specify new paragaphs other than making a gab in the code
\usepackage[titletoc,title]{appendix} % For the appendix in the end
\usepackage{afterpage} % Flushes unprocessed floats between pages, an optimization thing
\usepackage{amsmath} % Math shit
\usepackage{url} % For url formatation
\usepackage{tikz} % To draw things in latex, lets be real... you prob wont use this.. XD but now its in
\usepackage{amsfonts} % For the extra math symbols
\usepackage{lipsum} % For when you don't wanna write your own bullshit and wanna paste the standard, \lipsum
%\usepackage[danish]{babel} % For the dansish date formatting, datetime overrides this
\usepackage[ddmmyyyy]{datetime} % For nice date date formatting
\usepackage{comment} % For out-commenting everything between \begin{comment} something... \end{comment}
\usepackage{hyperref} % Enhanced references

% Provides the gnatt chart for project planning, useful and looks good
\usepackage{pgfgantt}
\usepackage{resources/pgfgnatt-custom}

\usepackage[newfloat]{minted} % For importing code
%\begin{code}
%\captionof{listing}{name of file or code}
%\label{code:myreferencelabel}
%    \begin{minted}{c}
%    code here
%    \end{minted}
%\end{code}

\usepackage{animate} % SUPER USEFUL AND NICE ANIMATION IN PAPERS!
% A bit hard... well... I think the appropriate word is tedious to use, but it looks AMAZING
% See example in main