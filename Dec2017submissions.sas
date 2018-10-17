libname text 'C:\Users\Evan\Desktop\Evan MSA\Fall 2\Text Mining';
run;

/*importing the .txt file that is in the Google Drive
Adding each line as one observation, max length at 500 charachters*/

data text;
	infile "C:\Users\Evan\Desktop\Evan MSA\Fall 2\Text Mining\testfile-2017-12.txt";
	input @1 sub $500.;
run;

/*Removing trailing blanks in the submission, and converting utc to the 
equivalent SAS numeric date*/
/*Reddit crosspost format is different, and this doesn't pick up the date for crossposts
which makes for an easy way to filter out crossposts (duplicate submissions)*/
data text;
	set text;
	stripped=strip(sub);
	submission=substr(stripped,1,length(stripped)-10);
	utc = SUBSTR(stripped,LENGTH(stripped)-9);
	utcnum=utc+315619200;
run;

/*Converting the date to a readable format*/
/*Likely a much cleaner way to do this, and more useful format!*/
data textdate;
	set text;
	date_var=put(utcnum, b8601dz20.);
run;

/*Sorting by date*/
proc sort data=textdate out=textdatesort;
	by date_var;
run;


/*Removes everything not from December 2017, which due to formatting differences,
gets rid of all crossposts (duplicate submissions)*/
data text.decbtc;
	set textdatesort;
	where date_var ge '20171201T000000+0000';
	keep submission date_var;
run;


/*Can use this to split into interesting dates/times corresponding to price peaks/floors
proc sql;
	select submission, date_var
	from textdate
	where date_var le '20171204T111952+0000';
run;
quit; 

