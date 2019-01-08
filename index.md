## Welcome to GitHub Pages

### Schedule


This schedule is still under development and is subject to change.  





<table class="table table-striped">
   <colgroup>
      <col class="col-md-1">
      <col class="col-md-1">
      <col class="col-md-2">
      <col class="col-md-8">
   </colgroup>
<thead>
   <tr>
      <th> Week </th>
      <th> Lecture </th>
      <th> Date </th>
      <th> Topic </th>
   </tr>
</thead>
<tbody>


<!-- This is the dates for all the lectures -->
{% capture dates %}
 01/17/2017 01/19/2017
 01/24/2017 01/26/2017
 01/31/2017 02/02/2017
 02/07/2017 02/09/2017
 02/14/2017 02/16/2017
 02/21/2017 02/23/2017
 02/28/2017 03/02/2017
 03/07/2017 03/09/2017
 03/14/2017 03/16/2017
 03/21/2017 03/23/2017
 03/28/2017 03/30/2017
 04/04/2017 04/06/2017
 04/11/2017 04/13/2017
 04/18/2017 04/20/2017
 04/25/2017 04/27/2017
 05/02/2017 05/04/2017
 05/11/2017
{% endcapture %}
{% assign dates = dates | split: " " %}




{% include syllabus_entry dates=dates %}
### Lecture 1
{% include syllabus_entry end=true %}


{% include syllabus_entry dates=dates %}
### Lecture 2
{% include syllabus_entry end=true %}


{% include syllabus_entry dates=dates %}
### Lecture 3
{% include syllabus_entry end=true %}


{% include syllabus_entry dates=dates %}
### Lecture 4
{% include syllabus_entry end=true %}



</tbody>
</table>


### Rest of stuff



You can use the [editor on GitHub](https://github.com/ucbrise/294-aisys/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ucbrise/294-aisys/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
